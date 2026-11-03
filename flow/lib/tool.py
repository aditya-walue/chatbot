# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, validate_call

_PRIMITIVE_SCHEMAS: dict[type, dict[str, str]] = {
	str: {"type": "string"},
	int: {"type": "integer"},
	float: {"type": "number"},
	bool: {"type": "boolean"},
	bytes: {"type": "string"},
}


class Problem(str):
	"""What is wrong with a tool call, as a plain statement a person can read, with `advice` for
	the model on what to do next. The statement is the error shown to the user if the model
	cannot fix it; the advice goes only to the model, which keeps tool and function names out of
	what people read."""

	advice: str

	def __new__(cls, fact: str, advice: str = "") -> "Problem":
		problem = super().__new__(cls, fact)
		problem.advice = advice
		return problem


class ToolError(ValueError):
	"""A tool that cannot do what it was asked, raised with a plain statement a person can read and
	`advice` for the model (see Problem, for refusals decided before a tool runs)."""

	def __init__(self, fact: str, advice: str = ""):
		super().__init__(fact)
		self.advice = advice


def problem_payload(problem: str | BaseException) -> dict[str, str]:
	"""The tool result for a problem or ToolError: {"error": statement[, "next": advice]}."""
	advice = getattr(problem, "advice", "")
	return {"error": str(problem), **({"next": advice} if advice else {})}


@dataclass(frozen=True)
class LayaSignals:
	"""What Laya said about a message besides its intent, for a tool deciding whether to take it."""

	asks_count: float = 0.0
	threshold: float = 0.0


@dataclass(frozen=True)
class LayaRoute:
	"""A tool's claim on a message Laya classified as `intent`: `arguments` reads the message and
	returns the call's arguments, or None to pass. Tools are asked in registry order, so one that
	answers a narrower request is listed before one that answers a broader one. Read-only tools only."""

	intent: str
	arguments: Callable[[str, LayaSignals], dict[str, Any] | None]


@dataclass
class Tool:
	name: str
	description: str
	parameters: dict[str, Any]
	func: Callable[..., Any]
	requires_confirmation: bool = False
	confirm_prompt: Callable[[dict[str, Any]], str] | None = None
	# Returns an error for arguments that can't work, checked before the user is asked to
	# approve — so an empty or malformed write is bounced back to the model, not to the user.
	# May normalise the arguments in place (e.g. wrap a lone object in a list). `context` is
	# the conversation's user and tool text, for checking values weren't made up; None when
	# the user has already approved the call.
	precheck: Callable[[dict[str, Any], str | None], str | None] | None = None
	# The tool's result carries a finished reply in its "answer" key, shown to the user as-is;
	# the run ends there instead of asking the model to restate it (small models garble copies).
	final_answer: bool = False
	# Re-routes a call the model sent to the wrong tool: returns (tool name, arguments) to run
	# instead, or None. E.g. an update of records that don't exist is a create.
	redirect: Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None] | None = None
	# Recognises a user message this tool unmistakably answers ("how many employees?") and
	# returns the arguments to call it with, or None. Routed calls skip the model's guess.
	route: Callable[[str], dict[str, Any] | None] | None = None
	# Lets the optional Laya classifier send a message here (see LayaRoute).
	laya: LayaRoute | None = None
	# Name of an argument whose value must have been mentioned in the conversation, or be the record
	# type of the page the user has open, when the model chose the call itself: a small model asked
	# "what is this?" otherwise fills it with an example from the tool's description.
	named_in_conversation: str | None = None

	def __post_init__(self) -> None:
		self._validated = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(self.func)
		self._accepts_any_kwargs = any(
			p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(self.func).parameters.values()
		)

	def __call__(self, **kwargs: Any) -> Any:
		# Name the valid parameters: pydantic's "Unexpected keyword argument" leaves a model
		# that invented an argument nothing to correct with.
		allowed = self.parameters.get("properties") or {}
		unknown = sorted(set(kwargs) - set(allowed))
		if unknown and allowed and not self._accepts_any_kwargs:
			raise TypeError(
				f"{self.name} has no parameter(s) {', '.join(unknown)}. "
				f"Valid parameters: {', '.join(allowed)}."
			)
		return self._validated(**kwargs)

	def to_dict(self) -> dict[str, Any]:
		return {
			"type": "function",
			"function": {
				"name": self.name,
				"description": self.description,
				"parameters": self.parameters,
			},
		}


def tool(
	func: Callable[..., Any] | None = None,
	*,
	name: str | None = None,
	description: str | None = None,
	requires_confirmation: bool = False,
	confirm_prompt: Callable[[dict[str, Any]], str] | None = None,
	precheck: Callable[[dict[str, Any], str | None], str | None] | None = None,
	final_answer: bool = False,
	redirect: Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None] | None = None,
	route: Callable[[str], dict[str, Any] | None] | None = None,
	laya: LayaRoute | None = None,
	named_in_conversation: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
	def wrap(f: Callable[..., Any]) -> Tool:
		if not callable(f):
			raise TypeError("@tool can only be applied to callables")
		return Tool(
			name=name or f.__name__,
			description=description or (inspect.getdoc(f) or "").strip(),
			parameters=build_schema(f),
			func=f,
			requires_confirmation=requires_confirmation,
			confirm_prompt=confirm_prompt,
			precheck=precheck,
			final_answer=final_answer,
			redirect=redirect,
			route=route,
			laya=laya,
			named_in_conversation=named_in_conversation,
		)

	return wrap(func) if func is not None else wrap


def build_schema(func: Callable[..., Any]) -> dict[str, Any]:
	sig = inspect.signature(func)
	hints = get_type_hints(func, include_extras=True)

	properties: dict[str, dict[str, Any]] = {}
	required: list[str] = []

	for param_name, param in sig.parameters.items():
		if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
			continue
		if param_name in ("self", "cls"):
			continue

		hint = hints.get(param_name, Any)
		actual_type, description = _unwrap_annotated(hint)
		actual_type, is_optional = _unwrap_optional(actual_type)

		prop_schema = _type_to_schema(actual_type)
		if description:
			prop_schema["description"] = description

		properties[param_name] = prop_schema
		if param.default is inspect.Parameter.empty and not is_optional:
			required.append(param_name)

	schema: dict[str, Any] = {
		"type": "object",
		"properties": properties,
		"additionalProperties": False,
	}
	if required:
		schema["required"] = required
	return schema


def _unwrap_annotated(t: Any) -> tuple[Any, str | None]:
	metadata = getattr(t, "__metadata__", None)
	if metadata is None:
		return t, None
	actual = getattr(t, "__origin__", t)
	description = next((m for m in metadata if isinstance(m, str)), None)
	return actual, description


def _unwrap_optional(t: Any) -> tuple[Any, bool]:
	origin = get_origin(t)
	if origin not in (Union, types.UnionType):
		return t, False

	args = get_args(t)
	if type(None) not in args:
		return t, False

	remaining = tuple(a for a in args if a is not type(None))
	if len(remaining) == 1:
		return remaining[0], True
	if not remaining:
		return Any, True
	return Union[remaining], True


def _type_to_schema(t: Any) -> dict[str, Any]:
	if t is Any or t is object or t is None:
		return {}

	if t in _PRIMITIVE_SCHEMAS:
		return dict(_PRIMITIVE_SCHEMAS[t])

	if t is dict:
		return {"type": "object"}
	if t is list:
		return {"type": "array"}

	origin = get_origin(t)
	args = get_args(t)

	if origin in (list, tuple, set, frozenset, typing.List, typing.Tuple, typing.Set, typing.FrozenSet):
		items_schema = _type_to_schema(args[0]) if args else {}
		return {"type": "array", "items": items_schema}

	if origin in (dict, typing.Dict):
		return {"type": "object"}

	if origin is Literal:
		return _enum_schema(list(args))

	if origin in (Union, types.UnionType):
		return {"anyOf": [_type_to_schema(a) for a in args]}

	if isinstance(t, type):
		if issubclass(t, BaseModel):
			return _inline_pydantic_refs(t.model_json_schema())
		if issubclass(t, Enum):
			return _enum_schema([m.value for m in t])

	return {}


def _inline_pydantic_refs(schema: dict[str, Any]) -> dict[str, Any]:
	defs = schema.pop("$defs", {})
	if not defs:
		return schema

	resolved: dict[str, dict[str, Any]] = {}

	def walk(node: Any) -> Any:
		if isinstance(node, list):
			return [walk(item) for item in node]
		if not isinstance(node, dict):
			return node
		ref = node.get("$ref")
		if isinstance(ref, str) and ref.startswith("#/$defs/"):
			def_name = ref.split("/")[-1]
			if def_name in resolved:
				return resolved[def_name]
			if def_name in defs:
				return walk(defs[def_name])
			return {"type": "object"}
		return {key: walk(value) for key, value in node.items()}

	for def_name, def_schema in defs.items():
		resolved[def_name] = walk(def_schema)

	return walk(schema)


def _enum_schema(values: list[Any]) -> dict[str, Any]:
	value_types = {type(v) for v in values}
	if value_types == {str}:
		return {"type": "string", "enum": values}
	if value_types == {bool}:
		return {"type": "boolean", "enum": values}
	if value_types <= {int}:
		return {"type": "integer", "enum": values}
	if value_types <= {int, float}:
		return {"type": "number", "enum": values}
	return {"enum": values}
