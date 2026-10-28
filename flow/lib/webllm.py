# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Browser-run models (WebLLM).

A Flow Model whose id starts with `webllm/` is not called from the server. Each model
call is handed to the chat panel that started the run: the SSE stream carries a
`BrowserRequest` with plain chat messages, the panel runs the model locally on WebGPU
(@mlc-ai/web-llm), and posts the raw reply back through `submit_browser_reply`. The
server side of the stream waits for that reply, then parses it into a ChatResponse so
the agent loop — tools, approvals, persistence — runs exactly as for a hosted model.

WebLLM has no reliable native tool calling across models, so tools are described in
the system prompt and called with `<tool_call>{json}</tool_call>` blocks (the Hermes /
Qwen 2.5 convention). The stored transcript keeps the standard OpenAI shape; the
plain-text rendering here is only what the browser model sees.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import frappe
from frappe import _

from flow.lib.model import ChatResponse, ToolCall, _build_tool_call

BROWSER_PROVIDER = "webllm"
# Name given to a tool call whose block couldn't be parsed (see to_plain_messages).
INVALID_CALL = "invalid_tool_call"
# Keep in sync with DEFAULT_CONTEXT_WINDOW in frontend/src/lib/webllm.js.
DEFAULT_CONTEXT_WINDOW = 16384
MAX_BROWSER_TOOL_RESULT = 3000

# The first call in a browser downloads the model weights (GBs), so allow minutes.
REPLY_TIMEOUT = 900
POLL_INTERVAL = 0.2
# Keeps the otherwise idle SSE connection alive through proxies while the browser works.
HEARTBEAT_INTERVAL = 15

# A block ends at its closing tag — or, when the model runs on without one, at the next call or
# at a <tool_response> it starts to invent.
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*(?=</tool_call>|<tool_call>|<tool_response|$)", re.DOTALL)
# An unquoted object key: `name:` (only treated as a key right after `{` or `,`).
BARE_KEY_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:")


@dataclass
class BrowserRequest:
	"""Stream event: run one model call in the browser and post the reply back under `id`.
	`response_schema`, when set, is a JSON schema the browser enforces while decoding."""

	id: str
	model: str
	messages: list[dict[str, str]]
	params: dict[str, Any] = field(default_factory=dict)
	response_schema: str | None = None
	# A call made for the server's own use (a yes / no question): the panel runs it but does not
	# show the model's text, which would otherwise appear in the chat before the real answer.
	silent: bool = False


@dataclass
class Heartbeat:
	"""Stream event with no content, sent while waiting on the browser."""


def strict_json_enabled() -> bool:
	"""Strict mode (site config `flow_webllm_strict_json`, on by default): the model answers with
	one JSON object per step, enforced while decoding by a schema, instead of free text with
	<tool_call> blocks — so a tool call can't be malformed, name an unknown tool, or pass an
	argument the tool doesn't take."""
	return bool(frappe.conf.get("flow_webllm_strict_json", True))


def response_schema(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
	"""One object: {"reply": text}, or {"tool": <offered tool>, "arguments": <its parameters>}."""
	options: list[dict[str, Any]] = [
		{
			"type": "object",
			"properties": {"reply": {"type": "string"}},
			"required": ["reply"],
			"additionalProperties": False,
		}
	]
	for tool in tools or []:
		fn = tool["function"]
		options.append(
			{
				"type": "object",
				"properties": {
					"tool": {"type": "string", "enum": [fn["name"]]},
					"arguments": fn.get("parameters") or {"type": "object"},
				},
				"required": ["tool", "arguments"],
				"additionalProperties": False,
			}
		)
	return {"anyOf": options}


def is_browser_model(model_id: str | None) -> bool:
	return (model_id or "").split("/", 1)[0] == BROWSER_PROVIDER


def browser_context_window(params: dict[str, Any]) -> int:
	"""The context window the panel loads a browser model with: the model's
	`context_window_size` param, else the panel default (frontend/src/lib/webllm.js)."""
	return int(params.get("context_window_size") or DEFAULT_CONTEXT_WINDOW)


def preload_hint(model_id: str | None) -> dict[str, Any] | None:
	"""What the panel needs to load a browser model before its first real call (so a greeting
	can start the weights download): the model name WebLLM knows and the context window the
	run will use. None when `model_id` isn't an enabled browser model."""
	if not is_browser_model(model_id):
		return None
	row = frappe.db.get_value("Flow Model", {"model_id": model_id, "enabled": 1}, ["name", "params"])
	if not row:
		return None
	params = row[1]
	return {
		"model": model_id.split("/", 1)[1],
		"context_window_size": browser_context_window(json.loads(params) if params else {}),
	}


def chat_stream(
	model_id: str,
	messages: list[dict[str, Any]],
	tools: list[dict[str, Any]] | None,
	params: dict[str, Any] | None,
	choices: list[str] | None = None,
) -> Generator[BrowserRequest | Heartbeat, None, ChatResponse]:
	"""Hand one model call to the browser and wait for its reply. Same contract as
	`Model.chat(stream=True)`: yields stream items, returns the ChatResponse. With `choices`, the
	call is a hidden classification: the browser must answer with one of them (enforced while
	decoding), and the ChatResponse content is that word."""
	request_id = frappe.generate_hash(length=20)
	frappe.cache.set_value(
		_owner_key(request_id), frappe.session.user, expires_in_sec=REPLY_TIMEOUT + 60
	)
	strict = strict_json_enabled()
	yield BrowserRequest(
		id=request_id,
		model=model_id.split("/", 1)[1],
		messages=to_plain_messages(messages, tools, strict=strict and not choices),
		params=params or {},
		response_schema=json.dumps(_choice_schema(choices)) if choices else (json.dumps(response_schema(tools)) if strict else None),
		silent=bool(choices),
	)

	reply = yield from _wait_for_reply(request_id)
	if reply.get("error"):
		raise RuntimeError(reply["error"])
	if choices:
		return ChatResponse(content=_chosen(reply.get("content") or "", choices), usage=reply.get("usage") or {})
	return parse_reply(reply.get("content") or "", reply.get("usage") or {})


def _choice_schema(choices: list[str]) -> dict[str, Any]:
	return {
		"type": "object",
		"properties": {"verdict": {"type": "string", "enum": choices}},
		"required": ["verdict"],
		"additionalProperties": False,
	}


def _chosen(content: str, choices: list[str]) -> str:
	"""The word the browser chose from `choices`, or "" when its reply is none of them."""
	try:
		word = str(json.loads(content).get("verdict", ""))
	except (ValueError, AttributeError):
		word = content.strip().strip('"{}')
	return word if word in choices else ""


def submit_reply(request_id: str, reply: dict[str, Any]) -> bool:
	"""Store the browser's reply for the waiting stream; False if nothing is waiting any more.

	A reply can outlive its request harmlessly — the run was stopped, timed out, or the server
	restarted mid-turn — so that is reported quietly rather than raised (which the desk would
	show as an error dialog). Answering another user's request is still refused."""
	owner = frappe.cache.get_value(_owner_key(request_id), use_local_cache=False)
	if owner is None:
		return False
	if owner != frappe.session.user:
		frappe.throw(_("Not permitted to answer this model request."), frappe.PermissionError)
	frappe.cache.set_value(_reply_key(request_id), reply, expires_in_sec=300)
	return True


def _wait_for_reply(request_id: str) -> Generator[Heartbeat, None, dict[str, Any]]:
	key = _reply_key(request_id)
	deadline = time.monotonic() + REPLY_TIMEOUT
	next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL
	try:
		while time.monotonic() < deadline:
			# Bypass the per-request local cache: the reply lands from another request.
			reply = frappe.cache.get_value(key, use_local_cache=False)
			if reply is not None:
				return reply
			if time.monotonic() >= next_heartbeat:
				next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL
				yield Heartbeat()
			time.sleep(POLL_INTERVAL)
	finally:
		frappe.cache.delete_value([key, _owner_key(request_id)])
	raise TimeoutError(_("The browser model did not reply in time."))


def to_plain_messages(
	messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, strict: bool = False
) -> list[dict[str, str]]:
	"""Render an OpenAI-style transcript as system/user/assistant text messages, with the
	tool schemas in the system prompt and tool calls/results as tagged text. Consecutive
	messages of the same role are merged, since chat templates expect turns to alternate."""
	system_parts: list[str] = []
	out: list[dict[str, str]] = []

	def add(role: str, content: str) -> None:
		if out and out[-1]["role"] == role:
			out[-1]["content"] = f"{out[-1]['content']}\n\n{content}".strip()
		else:
			out.append({"role": role, "content": content})

	# A call that couldn't be parsed is shown back as a plain note, never as a call to a tool
	# named "invalid_tool_call" — a small model copies that example and keeps calling it.
	malformed = {
		tc["id"]
		for m in messages
		if m.get("role") == "assistant"
		for tc in m.get("tool_calls") or []
		if tc["function"]["name"] == INVALID_CALL
	}
	for message in messages:
		role = message.get("role")
		content = _text(message.get("content"))
		if role == "system":
			system_parts.append(content)
		elif role == "user":
			add("user", content)
		elif role == "assistant":
			calls = [
				_render_call(tc["function"]["name"], json.loads(tc["function"].get("arguments") or "{}"), strict)
				for tc in message.get("tool_calls") or []
				if tc["id"] not in malformed
			]
			if strict and not calls and content:
				content = json.dumps({"reply": content}, ensure_ascii=False)
			text = "\n".join(filter(None, [content if not (strict and calls) else "", *calls]))
			if text:
				add("assistant", text)
		elif role == "tool" and message.get("tool_call_id") in malformed:
			add("user", f"(Your last tool call could not be read: {_error_text(content)})")
		elif role == "tool":
			tool_res = content
			if len(tool_res) > MAX_BROWSER_TOOL_RESULT:
				tool_res = tool_res[:MAX_BROWSER_TOOL_RESULT] + "\n...(truncated)"
			add("user", f"Tool result:\n{tool_res}" if strict else f"<tool_response>\n{tool_res}\n</tool_response>")

	if tools:
		system_parts.append(_strict_tools_prompt(tools) if strict else _tools_prompt(tools))
	if system_parts:
		out.insert(0, {"role": "system", "content": "\n\n".join(p for p in system_parts if p)})
	return out


def _parse_strict(text: str, usage: dict[str, Any]) -> ChatResponse | None:
	stripped = (text or "").strip()
	if not stripped.startswith("{") or "<tool_call>" in stripped:
		return None
	payload = _loads_lenient(stripped)
	if not isinstance(payload, dict):
		return None
	usage = {k: int(usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
	if isinstance(payload.get("tool"), str):
		arguments = payload.get("arguments")
		raw = arguments if isinstance(arguments, str) else json.dumps(arguments if arguments is not None else {})
		call = _build_tool_call(f"call_{frappe.generate_hash(length=12)}", payload["tool"], raw)
		return ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls", usage=usage)
	if isinstance(payload.get("reply"), str):
		return ChatResponse(content=payload["reply"].strip() or None, tool_calls=[], finish_reason="stop", usage=usage)
	return None


def _error_text(content: str) -> str:
	try:
		return json.loads(content).get("error") or content
	except (ValueError, AttributeError):
		return content


def parse_reply(text: str, usage: dict[str, Any]) -> ChatResponse:
	"""Read a browser model's raw reply: a strict-mode JSON object ({"reply"} or {"tool",
	"arguments"}), or prose with `<tool_call>` blocks."""
	strict = _parse_strict(text, usage)
	if strict is not None:
		return strict
	tool_calls: list[ToolCall] = []
	for raw in TOOL_CALL_PATTERN.findall(text):
		call_id = f"call_{frappe.generate_hash(length=12)}"
		payload = _loads_lenient(raw)
		if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
			problem = f"Malformed <tool_call> block: {raw[:200]}." if raw.strip() else "The <tool_call> block was empty."
			tool_calls.append(
				ToolCall(
					id=call_id,
					name=INVALID_CALL,
					arguments={},
					error=f"{problem} Write it as <tool_call>"
					'{"name": "<tool name>", "arguments": {...}}</tool_call>.',
				)
			)
			continue
		arguments = payload.get("arguments")
		if isinstance(arguments, str):
			raw_args = arguments
		else:
			raw_args = json.dumps(arguments if arguments is not None else {})
		tool_calls.append(_build_tool_call(call_id, payload["name"], raw_args))

	content = text.split("<tool_call>", 1)[0].strip() or None
	return ChatResponse(
		content=content,
		tool_calls=tool_calls,
		finish_reason="tool_calls" if tool_calls else "stop",
		usage={
			key: int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
		},
	)


def _loads_lenient(raw: str) -> Any:
	"""Parse a tool call's JSON, repairing the slips small models make: unbalanced brackets
	(`[{...}}}` with the list's `]` missing), unquoted keys (`{arguments: ...}`) and trailing
	commas — and trailing junk after a complete object (`{...}}��` when the model ran on).
	Returns None if still unparseable."""
	try:
		return json.loads(raw)
	except ValueError:
		pass
	first = _first_object(raw)
	if first is not None:
		return first
	try:
		return json.loads(_balance_brackets(_fix_keys_and_commas(raw)))
	except ValueError:
		return None


def _first_object(raw: str) -> Any:
	"""The complete JSON object at the start of `raw`, ignoring whatever follows it."""
	start = raw.find("{")
	if start == -1:
		return None
	try:
		value, _end = json.JSONDecoder().raw_decode(raw[start:])
	except ValueError:
		return None
	return value if isinstance(value, dict) else None


def _fix_keys_and_commas(raw: str) -> str:
	"""Quote bare object keys and drop trailing commas, leaving string contents untouched."""
	out: list[str] = []
	last = ""  # last non-whitespace character emitted
	i, n = 0, len(raw)
	while i < n:
		ch = raw[i]
		if ch == '"':
			end = _string_end(raw, i)
			out.append(raw[i:end])
			last = '"'
			i = end
			continue
		if ch == ",":
			nxt = raw[i + 1 :].lstrip()
			if nxt[:1] in ("}", "]"):
				i += 1
				continue
		match = BARE_KEY_PATTERN.match(raw, i)
		if match and last in ("{", ","):
			out.append(f'"{match.group(1)}"')
			last = '"'
			i = match.end(1)
			continue
		out.append(ch)
		if not ch.isspace():
			last = ch
		i += 1
	return "".join(out)


def _string_end(raw: str, start: int) -> int:
	"""Index just past the JSON string starting at `start` (or the end of `raw`)."""
	i = start + 1
	while i < len(raw):
		if raw[i] == "\\":
			i += 2
			continue
		if raw[i] == '"':
			return i + 1
		i += 1
	return len(raw)


def _balance_brackets(raw: str) -> str:
	"""Insert missing closers where a closer doesn't match the innermost open bracket, and
	append any still open at the end. Only closers are ever added; strings are skipped."""
	closer = {"{": "}", "[": "]"}
	stack: list[str] = []
	out: list[str] = []
	in_string = escaped = False
	for ch in raw:
		if in_string:
			if escaped:
				escaped = False
			elif ch == "\\":
				escaped = True
			elif ch == '"':
				in_string = False
		elif ch == '"':
			in_string = True
		elif ch in closer:
			stack.append(ch)
		elif ch in "}]":
			# Close inner brackets the model forgot until this closer matches one that's open.
			while stack and closer[stack[-1]] != ch and ch in (closer[b] for b in stack):
				out.append(closer[stack.pop()])
			if stack and closer[stack[-1]] == ch:
				stack.pop()
			else:
				continue  # stray closer with nothing to close: drop it
		out.append(ch)
	out.extend(closer[b] for b in reversed(stack))
	return "".join(out)


def _render_call(name: str, arguments: dict[str, Any], strict: bool) -> str:
	if strict:
		return json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)
	return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def _strict_tools_prompt(tools: list[dict[str, Any]]) -> str:
	lines = []
	for t in tools:
		fn = t["function"]
		name = fn["name"]
		desc = fn.get("description", "").strip()
		desc = desc.split("\n\n")[0].split("\n")[0].strip()
		params = fn.get("parameters") or {}
		props = params.get("properties") or {}
		trimmed_props = {}
		for k, v in props.items():
			prop_copy = dict(v)
			if "description" in prop_copy and isinstance(prop_copy["description"], str):
				prop_copy["description"] = prop_copy["description"].split(".")[0].strip()
			trimmed_props[k] = prop_copy
		req = params.get("required") or []
		req_suffix = f" Required: {json.dumps(req)}" if req else ""
		lines.append(f"- {name}: {desc} Arguments: {json.dumps(trimmed_props)}{req_suffix}")
	return (
		"# Tools\n\n"
		f"{chr(10).join(lines)}\n\n"
		"Answer with exactly one JSON object:\n"
		'- to use a tool: {"tool": "<name>", "arguments": {...}}\n'
		'- to answer the user: {"reply": "<your answer>"}\n'
		'A tool\'s output comes back as "Tool result:"; then reply or use another tool.'
	)


def _tools_prompt(tools: list[dict[str, Any]]) -> str:
	schemas = "\n".join(json.dumps(tool) for tool in tools)
	return (
		"# Tools\n\n"
		"You may call one or more functions to assist with the user query.\n\n"
		"You are provided with function signatures within <tools></tools> XML tags:\n"
		f"<tools>\n{schemas}\n</tools>\n\n"
		"For each function call, return a json object with function name and arguments "
		"within <tool_call></tool_call> XML tags:\n"
		'<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n'
		"Function results are given back to you within <tool_response></tool_response> tags."
	)


def _text(content: Any) -> str:
	"""Message content as text; multi-part content keeps only its text parts."""
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(
			part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
		)
	return str(content)


def _reply_key(request_id: str) -> str:
	return f"flow:webllm:reply:{request_id}"


def _owner_key(request_id: str) -> str:
	return f"flow:webllm:owner:{request_id}"
