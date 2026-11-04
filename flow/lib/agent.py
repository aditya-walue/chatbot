# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import frappe
from frappe.utils.caching import redis_cache
from frappe import _

from flow.lib import tracing
from flow.lib.model import ChatResponse, Model, ToolCall, ToolCallBegin
from flow.lib.tool import Problem, Tool, problem_payload

if TYPE_CHECKING:
	from flow.knowledge import Knowledge

DEFAULT_MAX_ITERATIONS = 20
MAX_FAILED_STEPS = 2
MAX_ERRORED_STEPS = 3  # steps in a row whose every tool call failed, before the user is told why
MAX_SAME_TOOL_STEPS = 3
RESEARCH_TEXT_LIMIT = 1500
STUCK_MESSAGE = (
	"Sorry, I couldn't put that request together correctly. Could you rephrase it, or say "
	"which record and fields you mean?"
)
ERROR_MESSAGE_LIMIT = 500
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass
class Question:
	"""An LLM-authored prompt shown to the user when a run pauses.

	The same shape covers a free-text ask (empty `options`), a single- or
	multi-select. When `allow_other` is true the picker always offers an "Other"
	choice that opens a textbox for the user to reiterate or redirect.
	`key` routes the answer back (e.g. the tool_call_id it belongs to).
	"""

	prompt: str
	options: list[str] = field(default_factory=list)
	multi_select: bool = False
	allow_other: bool = True
	key: str | None = None


@dataclass
class RunResult:
	output: str | None
	messages: list[dict[str, Any]]
	tool_calls: list[ToolCall] = field(default_factory=list)
	iterations: int = 0
	usage: dict[str, int] = field(default_factory=dict)
	paused: bool = False
	questions: list[Question] = field(default_factory=list)


@dataclass
class TextChunk:
	"""A token delta from the model. Emitted as the LLM streams its reply."""

	text: str


@dataclass
class ToolStarted:
	"""A tool call is about to execute. Lets the UI render a 'thinking' indicator. Emitted as soon
	as the model starts streaming the call, so `arguments` may still be empty at that point — the
	full arguments arrive on the matching ToolEnded."""

	id: str
	name: str
	arguments: dict[str, Any]


@dataclass
class ToolEnded:
	"""A tool call finished. `result` is the JSON-serialized return value."""

	id: str
	name: str
	result: str


@dataclass
class Done:
	"""Terminal event: the run finished (Completed or Paused)."""

	result: RunResult


Event = TextChunk | ToolStarted | ToolEnded | Done


class Agent:
	def __init__(
		self,
		*,
		model: Model | str,
		name: str = "agent",
		instructions: str | None = None,
		tools: list[Tool] | None = None,
		knowledge: Knowledge | list[Knowledge] | None = None,
		max_iterations: int = DEFAULT_MAX_ITERATIONS,
		auto_approve: bool = False,
	):
		if max_iterations < 1:
			raise ValueError("max_iterations must be at least 1")

		# Autonomous runs (triggers) auto-run confirmation tools; nobody is there to approve.
		self.auto_approve = auto_approve
		self.name = name
		self.model = Model(model) if isinstance(model, str) else model
		self.instructions = instructions
		self.tools = list(tools or [])
		# The desk page the user is on, {"doctype"[, "name"]}, set per turn by the session.
		self.page: dict[str, str] | None = None
		# Opik: the Flow Session and Flow Run a run belongs to, set per turn by the session.
		self.trace_context: dict[str, str] = {}
		self._trace: Any = tracing.NO_TRACE
		if knowledge:
			from flow.tools.builtins import bind_search_knowledge

			items = knowledge if isinstance(knowledge, list) else [knowledge]
			self.tools.append(bind_search_knowledge([k.name for k in items]))
		self.max_iterations = max_iterations

		self._tools_by_name: dict[str, Tool] = {}
		for tool in self.tools:
			if tool.name in self._tools_by_name:
				raise ValueError(f"Duplicate tool name: {tool.name!r}")
			self._tools_by_name[tool.name] = tool

	def run(self, input: str | list[dict[str, Any]], *, stream: bool = False) -> RunResult | Generator[Event]:
		"""Run the agent on `input`. With `stream=True`, returns a generator of `Event`s
		(text deltas, tool start/end markers, and a final `Done` carrying the `RunResult`)."""
		messages = self._build_initial_messages(input)
		if stream:
			return self._loop_stream(messages, route=True)
		return self._loop(messages, route=True)

	def resume(
		self,
		messages: list[dict[str, Any]],
		answers: dict[str, Any],
		*,
		stream: bool = False,
	) -> RunResult | Generator[Event]:
		"""Continue a run that paused on a question.

		`answers` maps each pending tool_call_id to the user's answer: "Approve" runs
		the tool, "Deny" records the rejection and stops the run, and any other free
		text is returned to the LLM as redirect feedback so it can adjust and retry.
		"""
		if stream:
			return self._resume_stream(messages, answers)
		messages, _ = self._prepare_resume(messages, answers)
		if _has_denial(answers):
			return self._stopped_result(messages)
		return self._loop(messages, self._answered_calls(messages))

	def _resume_stream(self, messages: list[dict[str, Any]], answers: dict[str, Any]) -> Generator[Event]:
		"""Stream a resume: first replay the just-resolved tool results so the UI can fill in
		the tool cards that were awaiting an answer, then continue the agent loop (or stop
		if the user denied)."""
		messages, resolved = self._prepare_resume(messages, answers)
		for call, content in resolved:
			yield ToolEnded(id=call.id, name=call.name, result=content)
		if _has_denial(answers):
			yield Done(result=self._stopped_result(messages))
			return
		yield from self._loop_stream(messages, self._answered_calls(messages))

	def _stopped_result(self, messages: list[dict[str, Any]]) -> RunResult:
		"""Terminal result for a run the user denied: pending calls are already resolved in
		`messages`, so end the turn here without another model call (no further tokens)."""
		return RunResult(
			output=None,
			messages=messages,
			tool_calls=self._answered_calls(messages),
			iterations=0,
		)

	def new_session(self, *, title: str | None = None) -> Any:
		"""Start a persisted conversation driven by this code agent (session's agent link is left empty)."""
		from flow.lib.session import new_session

		return new_session(self, title=title)

	def snapshot(self) -> dict[str, Any]:
		"""Config record stored on each Flow Run for traceability."""
		return {
			"title": self.name,
			"model": self.model.model_id,
			"instructions": self.instructions,
			"tools": [t.name for t in self.tools],
			"max_iterations": self.max_iterations,
		}

	def _prepare_resume(
		self, messages: list[dict[str, Any]], answers: dict[str, Any]
	) -> tuple[list[dict[str, Any]], list[tuple[ToolCall, str]]]:
		"""Append a tool result for each pending call. Returns the new messages plus the
		(call, content) pairs resolved, so a streaming resume can replay them as events."""
		_validate_messages(messages)
		messages = list(messages)
		pending = self._pending_calls(messages)
		if not pending:
			raise ValueError("No questions awaiting an answer in the provided messages")

		resolved: list[tuple[ToolCall, str]] = []
		for call in pending:
			answer = answers.get(call.id)
			tool = self._tools_by_name.get(call.name)
			if tool is not None and tool.requires_confirmation:
				content = self._resolve_confirmation(call, answer)
			else:
				content = _serialize_tool_result(answer)
			messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
			resolved.append((call, content))
		return messages, resolved

	def _resolve_confirmation(self, call: ToolCall, answer: Any) -> str:
		"""Run the tool if approved; deny if rejected; redirect with user feedback otherwise."""
		if answer == "Approve":
			# Resumed calls are re-read from the stored transcript, which keeps the model's
			# original arguments; normalise them again as at confirmation time.
			tool = self._tools_by_name[call.name]
			problem = tool.precheck(call.arguments, None) if tool.precheck else None
			if problem:
				return json.dumps(problem_payload(problem))
			result = self._run_tool(call)
			return _serialize_tool_result(result)
		if answer == "Deny":
			return json.dumps({"status": "denied", "message": "User denied this tool call."})
		# Free-text "Other"
		return json.dumps(
			{
				"status": "redirect",
				"message": "Tool not executed.",
				"user_feedback": answer,
				"instruction": "The user wants changes before this proceeds. Read their feedback carefully, adjust your approach, and try again.",
			}
		)

	def _loop(
		self,
		messages: list[dict[str, Any]],
		executed_calls: list[ToolCall] | None = None,
		*,
		route: bool = False,
	) -> RunResult:
		self._begin_trace(messages)
		try:
			result = self._run_loop(messages, executed_calls, route=route)
		except BaseException as e:
			self._trace.fail(e)
			raise
		finally:
			trace, self._trace = self._trace, tracing.NO_TRACE
		trace.finish(result.output, metadata=_result_metadata(result))
		return result

	def _run_loop(
		self,
		messages: list[dict[str, Any]],
		executed_calls: list[ToolCall] | None = None,
		*,
		route: bool = False,
	) -> RunResult:
		tool_schemas = [t.to_dict() for t in self.tools] or None
		executed_calls = executed_calls if executed_calls is not None else []
		usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

		routed = self._routed_response(messages) if route else None
		failed_streak = 0
		errored_streak, last_error = 0, None
		same_tool_streak, last_tools = 0, frozenset()
		for iteration in range(1, self.max_iterations + 1):
			if routed is not None:
				response, routed = routed, None
			else:
				context, tools = self._model_context(messages, tool_schemas)
				started = tracing.now()
				try:
					response = self.model.chat(context, tools=tools)
				except Exception as e:
					self._trace_model(context, tools, None, started, error=e)
					raise
				self._trace_model(context, tools, response, started)
			_accumulate_usage(usage_total, response.usage)
			messages.append(_assistant_message(response))

			if not response.tool_calls:
				_say_what_failed(response, messages)
				return RunResult(
					output=response.content,
					messages=messages,
					tool_calls=executed_calls,
					iterations=iteration,
					usage=usage_total,
				)

			questions: list[Question] = []
			results: list[tuple[ToolCall, Any]] = []
			for call in response.tool_calls:
				result = self._invoke(call, messages)
				if isinstance(result, Question):
					result.key = call.id
					questions.append(result)
					continue
				if isinstance(result, dict):
					result.pop("options", None)  # candidates for a browser model to choose between

				executed_calls.append(call)
				results.append((call, result))
				messages.append(
					{
						"role": "tool",
						"tool_call_id": call.id,
						"content": _serialize_tool_result(result),
					}
				)

			# Two steps in a row of nothing but malformed calls: the model is stuck; stop rather
			# than spend the remaining iterations repeating it.
			failed_streak = failed_streak + 1 if all(c.error for c in response.tool_calls) else 0
			answer = None if questions else self._final_answer(results)
			if answer is None and not questions and failed_streak >= MAX_FAILED_STEPS:
				answer = STUCK_MESSAGE
			# Step after step of tool calls that all fail (a model repeating a record type that does
			# not exist): say what went wrong now, rather than let it run out its iterations.
			step_error = _step_error(results)
			errored_streak, last_error = (errored_streak + 1, step_error) if step_error else (0, None)
			if answer is None and not questions and errored_streak >= MAX_ERRORED_STEPS:
				answer = f"That did not work: {last_error}"
			# The same tool step after step (a small model re-searching the docs with new words)
			# is a loop; once there is something to show, show it instead of searching again.
			step_tools = frozenset(c.name for c in response.tool_calls)
			same_tool_streak = same_tool_streak + 1 if step_tools == last_tools else 1
			last_tools = step_tools
			if answer is None and not questions and same_tool_streak >= MAX_SAME_TOOL_STEPS:
				answer = _research_answer(messages)
			if answer is not None:
				messages.append({"role": "assistant", "content": answer})
				return RunResult(
					output=answer,
					messages=messages,
					tool_calls=executed_calls,
					iterations=iteration,
					usage=usage_total,
				)

			if questions:
				return RunResult(
					output=response.content,
					messages=messages,
					tool_calls=executed_calls,
					iterations=iteration,
					usage=usage_total,
					paused=True,
					questions=questions,
				)

		# Out of steps: show the best docs or search result gathered so far rather than an error.
		answer = _research_answer(messages)
		if answer is None:
			raise RuntimeError(f"Agent {self.name!r} exceeded max_iterations ({self.max_iterations})")
		messages.append({"role": "assistant", "content": answer})
		return RunResult(
			output=answer,
			messages=messages,
			tool_calls=executed_calls,
			iterations=self.max_iterations,
			usage=usage_total,
		)

	def _loop_stream(
		self,
		messages: list[dict[str, Any]],
		executed_calls: list[ToolCall] | None = None,
		*,
		route: bool = False,
	) -> Generator[Event]:
		self._begin_trace(messages)
		finished = False
		try:
			for event in self._run_loop_stream(messages, executed_calls, route=route):
				if isinstance(event, Done):
					finished = True
					self._trace.finish(event.result.output, metadata=_result_metadata(event.result))
				yield event
		except BaseException as e:
			# GeneratorExit: the client went away mid-stream.
			if not finished:
				self._trace.fail(e)
			raise
		finally:
			self._trace = tracing.NO_TRACE

	def _run_loop_stream(
		self,
		messages: list[dict[str, Any]],
		executed_calls: list[ToolCall] | None = None,
		*,
		route: bool = False,
	) -> Generator[Event]:
		tool_schemas = [t.to_dict() for t in self.tools] or None
		executed_calls = executed_calls if executed_calls is not None else []
		usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

		planner = route and self._plans()
		routed = self._routed_response(messages, exact=not planner) if route else None
		# The planner stands in for the exact routes: it names the tool, and the model fills in
		# its arguments from a list of just that tool. Later steps see every tool again.
		planned = (yield from self._plan(messages)) if planner and routed is None else None
		first_schemas = [s for s in tool_schemas or [] if s["function"]["name"] == planned] or None
		if routed is None and planned is None and route and (yield from self._outside_scope(messages)):
			from flow.tools.web import OUT_OF_SCOPE

			yield TextChunk(text=OUT_OF_SCOPE)
			messages.append({"role": "assistant", "content": OUT_OF_SCOPE})
			yield Done(
				result=RunResult(output=OUT_OF_SCOPE, messages=messages, tool_calls=executed_calls, iterations=1, usage=usage_total)
			)
			return
		failed_streak = 0
		errored_streak, last_error = 0, None
		same_tool_streak, last_tools = 0, frozenset()
		for iteration in range(1, self.max_iterations + 1):
			if routed is not None:
				response, routed = routed, None
			else:
				response = yield from self._stream_model(
					*self._model_context(messages, first_schemas if iteration == 1 and first_schemas else tool_schemas)
				)
			_accumulate_usage(usage_total, response.usage)
			messages.append(_assistant_message(response))

			if not response.tool_calls:
				said = _say_what_failed(response, messages)
				if said:
					yield TextChunk(text=said)
				yield Done(
					result=RunResult(
						output=response.content,
						messages=messages,
						tool_calls=executed_calls,
						iterations=iteration,
						usage=usage_total,
					)
				)
				return

			questions: list[Question] = []
			results: list[tuple[ToolCall, Any]] = []
			for call in response.tool_calls:
				# Re-announce with the full arguments now that they've finished streaming, before the
				# tool runs — so the UI shows the arguments during execution, not only with the result.
				yield ToolStarted(id=call.id, name=call.name, arguments=call.arguments)
				result = self._invoke(call, messages)
				if isinstance(result, Question):
					result.key = call.id
					questions.append(result)
					yield ToolEnded(id=call.id, name=call.name, result="")
					continue
				if isinstance(result, dict) and result.get("options"):
					result = yield from self._pick_option(messages, result)

				executed_calls.append(call)
				results.append((call, result))
				serialized = _serialize_tool_result(result)
				messages.append({"role": "tool", "tool_call_id": call.id, "content": serialized})
				yield ToolEnded(id=call.id, name=call.name, result=serialized)

			# Two steps in a row of nothing but malformed calls: the model is stuck; stop rather
			# than spend the remaining iterations repeating it.
			failed_streak = failed_streak + 1 if all(c.error for c in response.tool_calls) else 0
			answer = None if questions else self._final_answer(results)
			if answer is None and not questions and failed_streak >= MAX_FAILED_STEPS:
				answer = STUCK_MESSAGE
			# Step after step of tool calls that all fail (a model repeating a record type that does
			# not exist): say what went wrong now, rather than let it run out its iterations.
			step_error = _step_error(results)
			errored_streak, last_error = (errored_streak + 1, step_error) if step_error else (0, None)
			if answer is None and not questions and errored_streak >= MAX_ERRORED_STEPS:
				answer = f"That did not work: {last_error}"
			# The same tool step after step (a small model re-searching the docs with new words)
			# is a loop; once there is something to show, show it instead of searching again.
			step_tools = frozenset(c.name for c in response.tool_calls)
			same_tool_streak = same_tool_streak + 1 if step_tools == last_tools else 1
			last_tools = step_tools
			if answer is None and not questions and same_tool_streak >= MAX_SAME_TOOL_STEPS:
				answer = _research_answer(messages)
			if answer is not None:
				yield TextChunk(text=answer)
				messages.append({"role": "assistant", "content": answer})
				yield Done(
					result=RunResult(
						output=answer,
						messages=messages,
						tool_calls=executed_calls,
						iterations=iteration,
						usage=usage_total,
					)
				)
				return

			if questions:
				yield Done(
					result=RunResult(
						output=response.content,
						messages=messages,
						tool_calls=executed_calls,
						iterations=iteration,
						usage=usage_total,
						paused=True,
						questions=questions,
					)
				)
				return

		# Out of steps: show the best docs or search result gathered so far rather than an error.
		answer = _research_answer(messages)
		if answer is None:
			raise RuntimeError(f"Agent {self.name!r} exceeded max_iterations ({self.max_iterations})")
		yield TextChunk(text=answer)
		messages.append({"role": "assistant", "content": answer})
		yield Done(
			result=RunResult(
				output=answer,
				messages=messages,
				tool_calls=executed_calls,
				iterations=self.max_iterations,
				usage=usage_total,
			)
		)

	def _redirect(self, call: ToolCall, tool: Tool, messages: list[dict[str, Any]] | None) -> Tool:
		"""Swap a call the model aimed at the wrong tool for the right one (see Tool.redirect),
		in the call and in the stored transcript — a resumed approval re-reads the transcript."""
		target = tool.redirect(call.arguments) if tool.redirect else None
		if not target or target[0] not in self._tools_by_name:
			return tool
		call.name, call.arguments = target
		for message in reversed(messages or []):
			for tc in message.get("tool_calls") or []:
				if tc["id"] == call.id:
					tc["function"] = {"name": call.name, "arguments": json.dumps(call.arguments)}
					return self._tools_by_name[call.name]
		return self._tools_by_name[call.name]

	def _final_answer(self, results: list[tuple[ToolCall, Any]]) -> str | None:
		"""The reply to show as-is when every tool called this step is a final-answer tool that
		returned one; None to hand the results back to the model as usual."""
		if not results:
			return None
		answers = []
		for call, result in results:
			tool = self._tools_by_name.get(call.name)
			if not (tool and tool.final_answer and isinstance(result, dict) and result.get("answer")):
				return None
			answers.append(result["answer"])
		return "\n\n".join(answers)

	def _model_context(
		self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]] | None
	) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
		"""What one model call sees. A small browser model reads the whole prompt on every step
		(seconds per thousand tokens), so it gets only the tools that fit the request and only
		recent turns; server models get everything, as before. The transcript and the tools the
		agent will run are untouched — this only trims the prompt."""
		from flow.lib.webllm import is_browser_model

		messages = _with_page_note(messages, self.page)
		if not is_browser_model(getattr(self.model, "model_id", None)):
			return messages, tool_schemas
		if not frappe.conf.get("flow_webllm_compact_context", True):
			return messages, tool_schemas
		return _recent_turns(messages), _relevant_tools(messages, tool_schemas)

	def _stream_model(
		self,
		messages: list[dict[str, Any]],
		tool_schemas: list[dict[str, Any]] | None,
		choices: list[str] | None = None,
	) -> Generator[Event, None, ChatResponse]:
		"""One streamed model call, relaying its events; returns the assembled response. With
		`choices` it is a hidden classification the model answers with one of those words."""
		started = tracing.now()
		try:
			chunks = (
				self.model.chat(messages, tools=tool_schemas, stream=True, choices=choices)
				if choices
				else self.model.chat(messages, tools=tool_schemas, stream=True)
			)
		except Exception as e:
			self._trace_model(messages, tool_schemas, None, started, error=e)
			raise
		# Tool calls are announced mid-stream (ToolCallBegin) so the UI shows the tool the moment
		# the model starts it, before its arguments finish streaming.
		try:
			while True:
				item = next(chunks)
				if isinstance(item, ToolCallBegin):
					yield ToolStarted(id=item.id, name=item.name, arguments={})
				elif isinstance(item, str):
					yield TextChunk(text=item)
				else:
					# Browser-model hand-off (flow.lib.webllm): relayed to the client as-is.
					yield item
		except StopIteration as e:
			self._trace_model(messages, tool_schemas, e.value, started)
			return e.value
		except Exception as e:
			self._trace_model(messages, tool_schemas, None, started, error=e)
			raise

	def _outside_scope(self, messages: list[dict[str, Any]]) -> Generator[Event, None, bool]:
		"""Whether the browser model says the latest message is not about this site or ERPNext /
		Frappe (a poem, a joke, general knowledge). A 3B model does not reliably follow "refuse
		anything else" buried in a long prompt, but answers one short IN / OUT question well, so
		it is asked that first. Only a clear OUT counts; anything else goes on to the answer.
		Skipped for server models and when the site sets `flow_scope_check` off."""
		from flow.lib.webllm import is_browser_model

		if not is_browser_model(getattr(self.model, "model_id", None)) or not frappe.conf.get("flow_scope_check", True):
			return False
		users = [str(m.get("content") or "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
		if not users:
			return False
		# A short follow-up ("In this site") is only clear beside what came before it.
		replies = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content")]
		earlier = f"Earlier message: {users[-2][:200]}\n" if len(users) > 1 else ""
		earlier += f"Earlier reply: {replies[-1][:200]}\n" if replies and len(users) > 1 else ""
		# A record type of this site named in the message ("where can I find the customer?") says
		# it is about the site, which a 3B model does not always see from the words alone.
		named = _named_doctypes(users[-1])
		# Two record types in one message ("assign an employee to a project") is about the site;
		# no model needs to be asked.
		if len(named) >= 2:
			return False
		hint = f"This site has a record type named {named[0]}, and the latest message mentions it.\n" if named else ""
		# A word from this site's own business record types ("leave", "stock", "invoice") in the
		# message says it is about the site, whatever a small model makes of the wording.
		if _uses_site_words(users[-1]):
			return False
		reply = yield from self._silent_choice(SCOPE_CHECK, f"{hint}{earlier}Latest message: {users[-1][:400]}", ["IN", "OUT"])
		return reply.strip().lower().startswith("out")

	def _silent_choice(self, instructions: str, question: str, choices: list[str]) -> Generator[Event, None, str]:
		"""A hidden one-word question to the browser model: its reply, never shown to the user."""
		check = [{"role": "system", "content": instructions}, {"role": "user", "content": question}]
		call = self._stream_model(check, None, choices)
		try:
			while True:
				event = next(call)
				if not isinstance(event, TextChunk):
					yield event
		except StopIteration as stop:
			return (stop.value.content or "") if stop.value else ""

	def _plans(self) -> bool:
		"""Whether the planner picks the tool, in place of the exact routes: only for a browser
		model, and only when the site turns `flow_planner` on: the 3B model picked worse than the
		exact routes did (definitions, "is there any X", dates, record types), so it is opt-in."""
		from flow.lib.webllm import is_browser_model

		return bool(self.tools) and is_browser_model(getattr(self.model, "model_id", None)) and bool(
			frappe.conf.get("flow_planner", False)
		)

	def _plan(self, messages: list[dict[str, Any]]) -> Generator[Event, None, str | None]:
		"""The planner: a hidden choice of the one tool whose description fits the latest message,
		read from the tools' own descriptions. None when no tool fits or the reply is unclear, so
		the model answers as usual."""
		users = [str(m.get("content") or "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
		if not users or messages[-1].get("role") != "user":
			return None
		listing = "\n".join(f"{t.name}: {_brief(t.description)}" for t in self.tools)
		earlier = f"Earlier message: {users[-2][:200]}\n" if len(users) > 1 else ""
		page = f"Open page: {self.page['doctype']}\n" if (self.page or {}).get("doctype") else ""
		names = [t.name for t in self.tools]
		reply = yield from self._silent_choice(
			PLANNER, f"{page}{earlier}Latest message: {users[-1][:400]}\n\nTools:\n{listing}", names + ["none"]
		)
		picked = reply.strip().split(" ")[0].strip(".:`\"'")
		if picked in names:
			self._trace_route("planner", picked, {})
			return picked
		return None

	def _pick_option(self, messages: list[dict[str, Any]], result: dict[str, Any]) -> Generator[Event, None, dict[str, Any]]:
		"""Let the browser model choose the answer among the docs sections a final-answer tool found
		(`result["options"]`): matching words finds candidates, but only reading the question tells
		which one answers it. "0" means none does. Anything unclear keeps the best match; server
		models and sites with `flow_docs_pick` off keep it too."""
		from flow.lib.webllm import is_browser_model

		options = result.pop("options", None)
		if not options or not is_browser_model(getattr(self.model, "model_id", None)) or not frappe.conf.get("flow_docs_pick", True):
			return result
		users = [str(m.get("content") or "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
		if not users:
			return result
		listing = "\n".join(f"{i}. {o['label']}" for i, o in enumerate(options, 1))
		reply = yield from self._silent_choice(
			DOCS_PICK, f"Question: {users[-1][:400]}\n\nSections:\n{listing}", [str(i) for i in range(len(options) + 1)]
		)
		picked = reply.strip().split(".")[0].strip()
		if picked == "0":
			from flow.tools.web import NOT_IN_DOCS

			return {**result, "answer": NOT_IN_DOCS}
		if picked.isdigit() and 1 <= int(picked) <= len(options):
			return {**result, "answer": options[int(picked) - 1]["answer"]}
		return result

	def _begin_trace(self, messages: list[dict[str, Any]]) -> None:
		"""Open the Opik trace for this run (a no-op unless tracing is on)."""
		self._trace = tracing.start(
			self.name,
			input=list(messages),  # a copy: the loop appends to `messages` while the run goes on
			thread_id=self.trace_context.get("session"),
			metadata={
				"model": getattr(self.model, "model_id", None),
				"run": self.trace_context.get("run"),
				"page": self.page,
			},
		)

	def _trace_model(
		self,
		context: list[dict[str, Any]],
		tools: list[dict[str, Any]] | None,
		response: ChatResponse | None,
		started: Any,
		*,
		error: BaseException | None = None,
	) -> None:
		model_id = getattr(self.model, "model_id", None)
		from flow.lib.webllm import is_browser_model

		usage = (response.usage if response else None) or {}
		self._trace.span(
			"model",
			type="llm",
			input={"messages": context},
			output=response and {"content": response.content, "tool_calls": [c.name for c in response.tool_calls]},
			started=started,
			model=model_id,
			provider="webllm" if is_browser_model(model_id) else None,
			usage={k: v for k, v in usage.items() if isinstance(v, int)},
			# The tools the model was offered: what to check when it picked the wrong one.
			metadata={"tools_offered": [t["function"]["name"] for t in tools or []]},
			error=error,
		)

	def _routed_response(self, messages: list[dict[str, Any]], *, exact: bool = True) -> ChatResponse | None:
		"""A tool call for a user message one of the tools unmistakably answers (see Tool.route),
		standing in for the model's first step."""
		last = messages[-1] if messages else {}
		if last.get("role") != "user" or not isinstance(last.get("content"), str):
			return None
		text = _with_referenced_doctype(last["content"], messages, self.page)
		# The open record's ID added to the text only answers a question that names its fields;
		# "why is this late?" is not a request to see the record (the model, told the page, is).
		page_id = (self.page or {}).get("name")
		added_id = bool(page_id) and page_id in text and page_id not in last["content"]
		for tool in self.tools if exact else []:
			if not tool.route:
				continue
			try:
				arguments = tool.route(text)
			except Exception:
				# A broken route must not fail the chat (the model answers instead), but not hide either.
				frappe.logger("flow", allow_site=True).warning("Route of %s failed", tool.name, exc_info=True)
				arguments = None
			if arguments and added_id and tool.name == "show_records" and not arguments.get("about"):
				continue
			if arguments:
				self._trace_route("exact route", tool.name, arguments)
				return _routed_call(tool.name, arguments)
		# A question about the record the user has open ("how much tax is on this?"): the words
		# that name its fields, read from that record's own labels, decide what to show.
		on_page = self._open_record_answer(text)
		if on_page:
			self._trace_route("open record", on_page.tool_calls[0].name, on_page.tool_calls[0].arguments)
			return on_page
		# No exact route: let the optional Laya classifier pick a read-only tool when it's
		# confident; it returns None (chat model as usual) when disabled, unsure or unavailable.
		from flow.lib import laya_router

		# Laya reads the record type from the page but not the record ID: an ID like AB-XYZ-00005
		# reads to the classifier like an error code.
		page_type = {"doctype": self.page["doctype"]} if (self.page or {}).get("doctype") else None
		laya_text = _with_referenced_doctype(last["content"], messages, page_type)
		laya = laya_router.route(laya_text, set(self._tools_by_name))
		if laya:
			self._trace_route("laya", *laya)
		return _routed_call(*laya) if laya else None

	def _trace_route(self, how: str, tool: str, arguments: dict[str, Any]) -> None:
		"""A span for a tool picked without the chat model: which route chose it, and with what."""
		self._trace.span(f"route: {how}", input={"tool": tool, "arguments": arguments}, started=tracing.now())

	def _open_record_answer(self, text: str) -> ChatResponse | None:
		"""show_records for the open record when the question names some of its fields and isn't
		asking to change anything; None otherwise (Laya / the model, told about the page)."""
		page = self.page or {}
		if not page.get("name") or "show_records" not in self._tools_by_name:
			return None
		from flow.tools.answers import WRITE_INTENT, asked_fields

		if WRITE_INTENT.search(text or ""):
			return None
		about = asked_fields(page["doctype"], text, exclude=page["name"], record=page["name"])
		if not about:
			return None
		return _routed_call("show_records", {"doctype": page["doctype"], "names": [page["name"]], "about": about})

	def _pending_calls(self, messages: list[dict[str, Any]]) -> list[ToolCall]:
		"""Tool calls in the transcript that have no tool result yet (awaiting an answer)."""
		return self._transcript_calls(messages, answered=False)

	def _answered_calls(self, messages: list[dict[str, Any]]) -> list[ToolCall]:
		"""Tool calls in the transcript that already have a tool result (executed)."""
		return self._transcript_calls(messages, answered=True)

	def _transcript_calls(self, messages: list[dict[str, Any]], *, answered: bool) -> list[ToolCall]:
		has_result = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
		calls: list[ToolCall] = []
		for message in messages:
			if message.get("role") != "assistant":
				continue
			for tc in message.get("tool_calls") or []:
				if (tc["id"] in has_result) != answered:
					continue
				fn = tc["function"]
				arguments = fn.get("arguments") or "{}"
				calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=json.loads(arguments)))
		return calls

	def _build_initial_messages(self, input: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""For a string, build [system?, user]. For a list, trust the caller and use it as-is —
		the caller owns the system message and any history."""
		if isinstance(input, str):
			messages: list[dict[str, Any]] = []
			if self.instructions:
				messages.append({"role": "system", "content": self.instructions})
			messages.append({"role": "user", "content": input})
			return messages
		_validate_messages(input)
		return list(input)

	def _invoke(self, call: ToolCall, messages: list[dict[str, Any]] | None = None) -> Any:
		"""Run a tool and return its raw result. A Question (returned or synthesized for
		`requires_confirmation` tools) signals a pause. `messages` is the transcript so far,
		for prechecks that verify argument values came from the user or a tool result."""
		if call.error:
			return json.dumps({"error": call.error})
		tool = self._tools_by_name.get(call.name)
		if tool is None:
			return json.dumps({"error": f"Unknown tool: {call.name!r}"})
		tool = self._redirect(call, tool, messages)
		context = _grounding_text(messages) if messages is not None else None
		problem = tool.precheck(call.arguments, context) if tool.precheck else None
		if not problem and messages is not None and not call.id.startswith(ROUTED_PREFIX):
			problem = _unmentioned(tool, call.arguments, messages, self.page)
		if problem:
			self._trace.span(call.name, type="tool", input=call.arguments, started=tracing.now(), error=problem)
			return json.dumps(problem_payload(problem))
		if tool.requires_confirmation and not self.auto_approve:
			return _confirmation_question(call, tool)
		return self._run_tool(call)

	def _run_tool(self, call: ToolCall) -> Any:
		"""Invoke the tool, returning its result or a serialized error message."""
		tool = self._tools_by_name[call.name]
		started = tracing.now()
		try:
			result = tool(**call.arguments)
		except Exception as e:
			self._trace.span(call.name, type="tool", input=call.arguments, started=started, error=e)
			payload = problem_payload(e)
			return json.dumps({**payload, "error": payload["error"][:ERROR_MESSAGE_LIMIT]})
		self._trace.span(call.name, type="tool", input=call.arguments, output=result, started=started)
		if tool.requires_confirmation:
			_save_write()
		return result


def _step_error(results: list[tuple[Any, Any]]) -> str | None:
	"""The last error of a step in which every tool call failed, else None."""
	errors = [_error_text(result) for _call, result in results]
	return errors[-1] if errors and all(errors) else None


def _error_text(result: Any) -> str | None:
	if isinstance(result, str):
		try:
			result = json.loads(result)
		except ValueError:
			return None
	error = result.get("error") if isinstance(result, dict) else None
	return error.strip() if isinstance(error, str) and error.strip() else None


def _say_what_failed(response: ChatResponse, messages: list[dict[str, Any]]) -> str | None:
	"""A model that ends its turn with no text right after a tool failed leaves the user with a red
	step and no reason: its reply becomes the tool's own error. Returns the text it added, if any."""
	if (response.content or "").strip():
		return None
	last = messages[-2] if len(messages) > 1 else {}
	if last.get("role") != "tool":
		return None
	try:
		error = json.loads(last.get("content") or "").get("error")
	except (ValueError, AttributeError):
		return None
	if not isinstance(error, str) or not error.strip():
		return None
	said = f"That did not work: {error.strip()}"
	response.content = said
	messages[-1] = {**messages[-1], "content": said}
	return said


def _save_write() -> None:
	"""Commit what an approved write tool just saved. A streamed response is iterated after the
	framework's own end-of-request commit, and the stream stays open while the browser model writes
	its reply (minutes while it loads), so the write would otherwise stay uncommitted, invisible
	to others and lost if the connection drops."""
	if not frappe.flags.in_test:
		frappe.db.commit()


# Tools that return schema and help text, not data: an option they list ("Gender: Male / Female
# / ...") is not a value the user chose.
METADATA_TOOLS = frozenset({"find_doctypes", "describe", "creation_steps"})


# Tools a request needs, by what it asks for (matched on the latest user message). The final-answer
# tools mostly run through exact routes; these let the model reach them for looser phrasings.
# Groups are ordered from specific to general so Qwen receives focused, non-overlapping tools.
TOOL_GROUPS: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = tuple(
	(re.compile(pattern, re.IGNORECASE), tools)
	for pattern, tools in (
		# Diagnose
		(r"\b(error|exception|traceback|fail(s|ed|ing)?|not working|(can'?t|cannot|won'?t|unable to)\s+(save|submit|create|cancel|open|login|log in|print))\b", ("error_diagnosis",)),
		# How-to guides: specific workflows
		(r"\b(assign|link|attach|allocat\w*)\b", ("assignment_steps", "explain_doctype")),
		(r"\b(how\b.*\b(mark|status|paid|submit|cancel|close)|mark\b.*\b(paid|status))\b", ("status_steps", "document_flow")),
		(r"\b(how (do|to|can|should)|steps?|guide|procedure)\b", ("docs_howto", "frappe_docs")),
		# Explain: roles and permissions
		(r"\b(role|roles|permission|permissions|access|access rights?|privilege)\b", ("doctype_access", "role_users", "user_roles", "role_permissions", "explain_doctype")),
		# Explain: document workflow & lifecycle
		(r"\b(after|next|process|flow|workflow|approv\w*|cycle|stage)\b", ("document_flow", "explain_doctype")),
		# Explain: DocType definition
		(r"\b(what (is|are|does)|explain|mean(s|ing)?|define|difference|purpose)\b", ("explain_doctype", "document_flow")),
		# Write tools: create and update
		(r"\b(create|add|make|new|register|enter|record a)\b", ("create", "required_values", "find_doctypes")),
		(r"\b(update|change|set|mark|close|rename|edit|modify)\b", ("update", "show_records", "find_doctypes")),
		# Records: count
		(r"\b(how many|count|number of)\b", ("count", "show_records")),
	)
)
# Looking things up is what the model falls back on when no specific intent matched.
DEFAULT_TOOLS = ("show_records", "count", "total", "low_stock", "on_leave", "where_to_find", "read", "find_doctypes")
MAX_TOOLS = 8
# Always offered beside the chosen tools: the fallback when none of them can answer.
FALLBACK_TOOLS = ("docs_howto", "frappe_docs", "web_search")
RECENT_USER_TURNS = 2


def _relevant_tools(
	messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
	if not tool_schemas:
		return tool_schemas
	text = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
	wanted: list[str] = []
	for pattern, tools in TOOL_GROUPS:
		if pattern.search(text):
			wanted.extend(tools)
			break
	if not wanted:
		wanted.extend(DEFAULT_TOOLS)
	# Tools already used this turn stay available, so a follow-up step can reuse them.
	turn_start = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
	for m in messages[turn_start:]:
		for tc in m.get("tool_calls") or []:
			wanted.insert(0, tc["function"]["name"])
	# small_talk is a route-only tool; never offer it to the model.
	by_name = {t["function"]["name"]: t for t in tool_schemas if t["function"]["name"] != "small_talk"}
	chosen = [by_name[n] for n in dict.fromkeys(wanted) if n in by_name][:MAX_TOOLS]
	chosen += [by_name[n] for n in FALLBACK_TOOLS if n in by_name and by_name[n] not in chosen]
	return chosen or tool_schemas


def _recent_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""The system prompt plus the last RECENT_USER_TURNS user turns (with their tool calls and
	results); older turns are dropped from the prompt, never from the stored transcript."""
	users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
	if len(users) <= RECENT_USER_TURNS:
		return messages
	start = users[-RECENT_USER_TURNS]
	system = [m for m in messages[:start] if m.get("role") == "system"]
	return system + messages[start:]


# The one question the browser model is asked before it answers a message no route took.
SCOPE_CHECK = (
	"You filter messages for an assistant that helps with a company's ERPNext / Frappe site. "
	"Answer IN or OUT. IN: the message is about business software or the site: its records, users, "
	"roles, permissions, settings, reports, errors, emails, notifications, alerts, reminders, "
	"where to find, open or see a record type of the site (customers, items, invoices), "
	"workflows, accounting, taxes (such as GST or VAT), invoices, stock, HR or sales, or about how "
	"to do or set up something in ERPNext / Frappe, or is a greeting, thanks, or a short follow-up "
	"to the earlier message. OUT: only what clearly has nothing to do with business software, such "
	"as a poem, a joke, general knowledge, maths, coding or career advice, weather, news or a "
	"translation. If unsure, answer IN."
)

PLANNER = (
	"You choose which tool answers the user's latest message. Each tool is listed as name: what it "
	"does. Reply with the name of the one tool whose description fits what the user asks for, or "
	"none if no tool fits. Choose by the meaning of the request, not by shared words."
)


def _brief(text: str, limit: int = 220) -> str:
	"""A tool's description cut to what a one-line listing can hold."""
	text = " ".join((text or "").split())
	return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# The question the browser model is asked to pick the docs section that answers a how-to.
DOCS_PICK = (
	"You pick which numbered documentation section answers the user's question. Answer with its "
	"number only. Answer 0 if none of them answers the question."
)

# Words of a message looked at for record types: each window is a lookup, and a long paste is no question.
MAX_NAMED_WORDS = 60
MIN_SITE_WORD = 4

# Ids of tool calls made by a route; a call with another id was chosen by the model.
ROUTED_PREFIX = "route_"


@redis_cache(ttl=3600)
def _site_vocabulary() -> list[str]:
	"""Every word of 4+ letters in the names of this site's business record types ("leave",
	"valuation", "invoice"), singular. Core Frappe's own types are left out, so its generic words
	("list", "about") don't count."""
	apps = dict(frappe.get_all("Module Def", fields=["name", "app_name"], as_list=True))
	words: set[str] = set()
	for name, module in frappe.get_all("DocType", filters={"istable": 0, "issingle": 0}, fields=["name", "module"], as_list=True):
		if apps.get(module) in (None, "frappe"):
			continue
		words.update(_singular(w) for w in name.lower().split() if len(w) >= MIN_SITE_WORD and w.isalpha())
	return sorted(words)


def _singular(word: str) -> str:
	return word[:-1] if len(word) > MIN_SITE_WORD and word.endswith("s") else word


def _uses_site_words(text: str) -> bool:
	"""Whether the message has a word from the site's business vocabulary."""
	letters = "".join(c if c.isalpha() else " " for c in (text or "").lower())
	vocabulary = set(_site_vocabulary())
	return any(_singular(w) in vocabulary for w in letters.split()[:MAX_NAMED_WORDS] if len(w) >= MIN_SITE_WORD)


def _named_doctypes(text: str) -> list[str]:
	"""The distinct record types of this site a message names, longest names first ("sales
	invoice" is one type, not "sales" and "invoice"); misspellings and plurals are matched as in
	routes."""
	from flow.tools.builtins import _doctype_from_phrase

	letters = "".join(c if c.isalpha() or c == "-" else " " for c in (text or "").lower())
	words = letters.split()[:MAX_NAMED_WORDS]
	used: set[int] = set()
	found: list[str] = []
	for size in range(min(4, len(words)), 0, -1):
		for start in range(len(words) - size + 1):
			span = set(range(start, start + size))
			if span & used:
				continue
			doctype = _doctype_from_phrase(" ".join(words[start : start + size]))
			if doctype:
				used |= span
				if doctype not in found:
					found.append(doctype)
	return found


def _unmentioned(tool: Tool, arguments: dict[str, Any], messages: list[dict[str, Any]], page: dict[str, str] | None) -> str | None:
	"""An error for a model-chosen call whose `tool.named_in_conversation` argument names something
	nobody mentioned (neither the user, a tool result, nor the open page), so the model asks
	instead of answering about a guess."""
	value = arguments.get(tool.named_in_conversation) if tool.named_in_conversation else None
	if not isinstance(value, str) or not value.strip():
		return None
	seen = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") in ("user", "tool")).lower()
	seen += f"\n{(page or {}).get('doctype', '')}".lower()
	words = [w for w in value.lower().replace("-", " ").split() if w]
	if all((w[:-1] if len(w) > 3 and w.endswith("s") else w) in seen for w in words):
		return None
	return Problem(
		f"{value!r} was not mentioned in this conversation.",
		f"Do not guess: ask the user which {tool.named_in_conversation} they mean, or use the one they named.",
	)


def _routed_call(name: str, arguments: dict[str, Any]) -> ChatResponse:
	"""A routed tool call, shaped like a model response that made it."""
	call = ToolCall(id=f"{ROUTED_PREFIX}{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)
	return ChatResponse(content=None, tool_calls=[call])


def _grounding_text(messages: list[dict[str, Any]]) -> str:
	"""Lower-cased text of what the user said and what data tools returned — the only
	legitimate sources for values the model writes into records."""
	tool_names = {
		tc["id"]: tc["function"]["name"]
		for m in messages
		if m.get("role") == "assistant"
		for tc in m.get("tool_calls") or []
	}
	return "\n".join(
		str(m.get("content") or "")
		for m in messages
		if m.get("role") == "user"
		or (m.get("role") == "tool" and tool_names.get(m.get("tool_call_id")) not in METADATA_TOOLS)
	).lower()


def _validate_messages(messages: Any) -> None:
	if not isinstance(messages, list):
		raise TypeError(f"input must be a str or list of message dicts, got {type(messages).__name__}")
	for i, message in enumerate(messages):
		if not isinstance(message, dict):
			raise TypeError(f"messages[{i}] must be a dict, got {type(message).__name__}")
		role = message.get("role")
		if role not in VALID_ROLES:
			raise ValueError(f"messages[{i}].role must be one of {sorted(VALID_ROLES)}, got {role!r}")
		if role == "tool" and not message.get("tool_call_id"):
			raise ValueError(f"messages[{i}] is a tool message but has no tool_call_id")
		if "content" not in message and "tool_calls" not in message:
			raise ValueError(f"messages[{i}] must have 'content' or 'tool_calls'")


def _assistant_message(response: ChatResponse) -> dict[str, Any]:
	message: dict[str, Any] = {"role": "assistant", "content": response.content}
	if response.tool_calls:
		message["tool_calls"] = [
			{
				"id": call.id,
				"type": "function",
				"function": {
					"name": call.name,
					"arguments": json.dumps(call.arguments),
				},
			}
			for call in response.tool_calls
		]
	return message


def _confirmation_question(call: ToolCall, tool: Tool) -> Question:
	"""Build the approval prompt shown to the user for a `requires_confirmation` tool call.
	Uses the tool's `confirm_prompt` for a plain-English summary, falling back to a JSON dump."""
	if tool.confirm_prompt:
		body = tool.confirm_prompt(call.arguments)
	else:
		body = json.dumps(call.arguments, indent=2, default=str)
	return Question(
		prompt=_("Approve `{0}`?\n\n{1}").format(call.name, body),
		options=["Approve", "Deny"],
		allow_other=True,
	)


def _has_denial(answers: dict[str, Any]) -> bool:
	"""A Deny answer halts the run — the user rejected an action, so stop rather than
	continue. Approve and free-text (redirect) answers let the run proceed."""
	return any(answer == "Deny" for answer in answers.values())


def _research_answer(messages: list[dict[str, Any]]) -> str | None:
	"""A reply built from the latest docs page or web results a lookup tool returned, or None when
	there is none in this turn. The text is the tool's own result, so it stays live data."""
	for message in reversed(messages):
		if message.get("role") == "user":
			break
		if message.get("role") != "tool":
			continue
		try:
			result = json.loads(message.get("content") or "")
		except ValueError:
			continue
		if not isinstance(result, dict):
			continue
		page = result.get("page")
		if isinstance(page, dict) and page.get("url") and (page.get("text") or page.get("snippet")):
			text = str(page.get("text") or page["snippet"])[:RESEARCH_TEXT_LIMIT].rstrip()
			return f"{text}\n\nSource: {page['url']}"
		found = [r for r in result.get("results") or [] if isinstance(r, dict) and r.get("url")]
		if found:
			lines = [f"- {r.get('title') or r['url']}: {r['url']}" for r in found]
			return "I found these pages that may help:\n\n" + "\n".join(lines)
	return None


def _serialize_tool_result(result: Any) -> str:
	if isinstance(result, str):
		return result
	if result is None:
		return ""
	try:
		return json.dumps(result, default=str)
	except (TypeError, ValueError):
		return str(result)


def _result_metadata(result: RunResult) -> dict[str, Any]:
	return {"iterations": result.iterations, "usage": result.usage, "paused": result.paused}


def _accumulate_usage(total: dict[str, int], delta: dict[str, int]) -> None:
	for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
		total[key] += int(delta.get(key, 0) or 0)


# "this invoice", "that order", "make it paid": the record on screen or from an earlier answer.
REFERENCE = re.compile(r"\b(this|that|these|those|it)\b(?:\s+([a-z][a-z\-]*))?", re.IGNORECASE)


def _with_referenced_doctype(
	text: str, messages: list[dict[str, Any]], page: dict[str, str] | None = None
) -> str:
	"""`text` with a reference spelled out, so routes and Laya can recognise it (the stored
	message is unchanged). The record open on the user's page comes first, then the DocType of
	the latest tool result:
	"how much tax is in this invoice" on an open Sales Invoice -> "... this Sales Invoice <ID>";
	"make this invoice paid" after a Sales Invoice was listed -> "make this Sales Invoice paid";
	"make it paid" -> "make the Sales Invoice paid". A word naming another record type ("that
	order" while on an invoice) is left alone."""
	match = REFERENCE.search(text or "")
	if not match:
		return _with_access_subject(text, messages, page)
	word = (match.group(2) or "").lower()
	for doctype, name in _referents(messages[:-1], page):
		names = doctype.lower().split()
		target = f"{doctype} {name}" if name else doctype
		if word and (word in names or word.rstrip("s") in names):
			return text[: match.start(2)] + target + text[match.end(2) :]
		if word and word.rstrip("s") in _doctype_words():
			continue  # names some other record type: try the next referent
		if match.group(1).lower() == "it":
			return text[: match.start(1)] + f"the {target}" + text[match.end(1) :]
		return text[: match.end(1)] + f" {target}" + text[match.end(1) :]
	# "…in this system" names no record, so it doesn't count as a reference.
	return _with_access_subject(text, messages, page)


def _with_access_subject(text: str, messages: list[dict[str, Any]], page: dict[str, str] | None) -> str:
	"""A question about who has access, that names no record type ("which user has the access in
	this system?" after one about Sales Invoice), with the DocType talked about last spelled out
	("... for Sales Invoice"); otherwise unchanged."""
	from flow.tools.answers import ACCESS_WORDS, WHO_ASKS, _builtins

	if not (WHO_ASKS.search(text or "") and ACCESS_WORDS.search(text or "")):
		return text
	if _builtins()._find_doctype_in_text(WHO_ASKS.sub(" ", text)):
		return text
	for doctype, _name in _referents(messages[:-1], page):
		return f"{text.rstrip(' ?.!')} for {doctype}"
	return text


def _referents(messages: list[dict[str, Any]], page: dict[str, str] | None) -> list[tuple[str, str]]:
	"""(DocType, record ID or "") a reference may mean: the open record first, then the latest
	tool result's DocType."""
	found = []
	if page and page.get("doctype"):
		found.append((page["doctype"], page.get("name") or ""))
	latest = _latest_doctype(messages)
	if latest and all(latest != doctype for doctype, _ in found):
		found.append((latest, ""))
	return found


def _with_page_note(messages: list[dict[str, Any]], page: dict[str, str] | None) -> list[dict[str, Any]]:
	"""The prompt with one system line saying which desk page the user has open, so "this" and
	questions without a record ID refer to it. The transcript itself is unchanged."""
	if not page or not page.get("doctype"):
		return messages
	where = f"{page['doctype']} {page['name']} open" if page.get("name") else f"the {page['doctype']} list open"
	note = f"The user has {where} in the desk; 'this' or 'it' usually means that."
	if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
		# Joined to the system prompt: trimmed browser prompts keep only the first system message.
		return [{**messages[0], "content": messages[0]["content"] + "\n\n" + note}, *messages[1:]]
	return [{"role": "system", "content": note}, *messages]


def _latest_doctype(messages: list[dict[str, Any]]) -> str | None:
	"""DocType of the latest tool result in the conversation that names one."""
	for message in reversed(messages):
		if message.get("role") != "tool" or not isinstance(message.get("content"), str):
			continue
		try:
			doctype = json.loads(message["content"]).get("doctype")
		except (ValueError, AttributeError):
			continue
		if isinstance(doctype, str) and doctype:
			return doctype
	return None


def _doctype_words() -> set[str]:
	"""Every word used in a DocType name, lower-cased ("sales", "invoice", "order", ...)."""
	from flow.tools.builtins import _doctype_names

	return {w for name in _doctype_names() for w in name.split()}
