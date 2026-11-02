# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import json
from typing import Any

import frappe
from frappe.tests import IntegrationTestCase, UnitTestCase

from flow.lib.agent import (
	Agent,
	Done,
	Question,
	RunResult,
	TextChunk,
	ToolEnded,
	ToolStarted,
)
from flow.lib.model import ChatResponse, ToolCall
from flow.lib.tool import tool


class FakeModel:
	def __init__(self, responses: list[ChatResponse], streams: list[list[str]] | None = None):
		"""`responses` are returned by chat(); `streams` (optional) are the text-delta
		sequences yielded when stream=True, one list per call, indexed alongside responses."""
		self._responses = list(responses)
		self._streams = list(streams) if streams is not None else [[r.content or ""] for r in responses]
		self.calls: list[dict[str, Any]] = []

	def chat(self, messages, tools=None, *, stream=False, choices=None):
		self.calls.append({"messages": list(messages), "tools": tools, "stream": stream})
		if not self._responses:
			raise AssertionError("FakeModel ran out of scripted responses")
		response = self._responses.pop(0)
		text_parts = self._streams.pop(0) if self._streams else [response.content or ""]
		if stream:
			return _scripted_stream(text_parts, response)
		return response


def _scripted_stream(text_parts: list[str], response: ChatResponse):
	for part in text_parts:
		if part:
			yield part
	return response


def _final(text: str, usage: dict[str, int] | None = None) -> ChatResponse:
	return ChatResponse(
		content=text,
		finish_reason="stop",
		usage=usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	)


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> ChatResponse:
	return ChatResponse(
		content=None,
		tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
		finish_reason="tool_calls",
		usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	)


class TestAgentBasics(UnitTestCase):
	def test_run_without_tools_returns_final_output(self):
		model = FakeModel([_final("hello")])
		agent = Agent(model=model)

		result = agent.run("hi")

		self.assertIsInstance(result, RunResult)
		self.assertEqual(result.output, "hello")
		self.assertEqual(result.iterations, 1)
		self.assertEqual(result.tool_calls, [])
		self.assertEqual(result.messages[0], {"role": "user", "content": "hi"})
		self.assertEqual(result.messages[-1]["role"], "assistant")

	def test_instructions_added_as_system_message(self):
		model = FakeModel([_final("ok")])
		agent = Agent(model=model, instructions="You are concise.")

		agent.run("hi")

		first_message = model.calls[0]["messages"][0]
		self.assertEqual(first_message, {"role": "system", "content": "You are concise."})

	def test_run_accepts_message_list_input(self):
		model = FakeModel([_final("ok")])
		agent = Agent(model=model)

		agent.run([{"role": "user", "content": "one"}, {"role": "user", "content": "two"}])

		messages = model.calls[0]["messages"]
		self.assertEqual([m["content"] for m in messages], ["one", "two"])

	def test_usage_is_accumulated_across_iterations(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel(
			[
				ChatResponse(
					content=None,
					tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
					usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
				),
				_final("done", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
			]
		)
		agent = Agent(model=model, tools=[ping])

		result = agent.run("hi")

		self.assertEqual(result.usage, {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11})


class TestAgentToolLoop(UnitTestCase):
	def test_executes_tool_call_and_loops_to_final_response(self):
		@tool
		def add(a: int, b: int) -> int:
			"""Add."""
			return a + b

		model = FakeModel([_tool_call("add", {"a": 2, "b": 3}), _final("result is 5")])
		agent = Agent(model=model, tools=[add])

		result = agent.run("add 2 and 3")

		self.assertEqual(result.output, "result is 5")
		self.assertEqual(result.iterations, 2)
		self.assertEqual(len(result.tool_calls), 1)
		self.assertEqual(result.tool_calls[0].name, "add")

		tool_message = next(m for m in result.messages if m["role"] == "tool")
		self.assertEqual(tool_message["tool_call_id"], "call_1")
		self.assertEqual(tool_message["content"], "5")

	def test_tool_schemas_are_passed_to_model(self):
		@tool
		def echo(text: str) -> str:
			"""Echo."""
			return text

		model = FakeModel([_final("ok")])
		agent = Agent(model=model, tools=[echo])

		agent.run("hi")

		passed_tools = model.calls[0]["tools"]
		self.assertEqual(len(passed_tools), 1)
		self.assertEqual(passed_tools[0]["function"]["name"], "echo")

	def test_no_tools_means_tools_param_is_none(self):
		model = FakeModel([_final("ok")])
		agent = Agent(model=model)

		agent.run("hi")

		self.assertIsNone(model.calls[0]["tools"])

	def test_unknown_tool_call_returns_error_to_model(self):
		model = FakeModel([_tool_call("nonexistent", {}), _final("recovered")])
		agent = Agent(model=model)

		result = agent.run("hi")

		tool_message = next(m for m in result.messages if m["role"] == "tool")
		payload = json.loads(tool_message["content"])
		self.assertIn("Unknown tool", payload["error"])
		self.assertEqual(result.output, "recovered")

	def test_tool_exception_is_caught_and_returned_to_model(self):
		@tool
		def boom() -> str:
			"""Raises."""
			raise RuntimeError("kaboom")

		model = FakeModel([_tool_call("boom", {}), _final("handled")])
		agent = Agent(model=model, tools=[boom])

		result = agent.run("hi")

		tool_message = next(m for m in result.messages if m["role"] == "tool")
		payload = json.loads(tool_message["content"])
		self.assertEqual(payload["error"], "kaboom")
		self.assertEqual(result.output, "handled")

	def test_tool_validation_error_is_caught(self):
		@tool
		def add(a: int, b: int) -> int:
			"""Add."""
			return a + b

		model = FakeModel([_tool_call("add", {"a": "not a number", "b": 3}), _final("recovered")])
		agent = Agent(model=model, tools=[add])

		result = agent.run("hi")

		tool_message = next(m for m in result.messages if m["role"] == "tool")
		self.assertIn("error", json.loads(tool_message["content"]))
		self.assertEqual(result.output, "recovered")

	def test_dict_tool_result_is_json_serialized(self):
		@tool
		def lookup(city: str) -> dict:
			"""Lookup."""
			return {"city": city, "temp": 72}

		model = FakeModel([_tool_call("lookup", {"city": "Mumbai"}), _final("ok")])
		agent = Agent(model=model, tools=[lookup])

		result = agent.run("hi")

		tool_message = next(m for m in result.messages if m["role"] == "tool")
		self.assertEqual(json.loads(tool_message["content"]), {"city": "Mumbai", "temp": 72})

	def test_multiple_tool_calls_in_one_response_all_execute(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel(
			[
				ChatResponse(
					content=None,
					tool_calls=[
						ToolCall(id="c1", name="ping", arguments={}),
						ToolCall(id="c2", name="ping", arguments={}),
					],
				),
				_final("done"),
			]
		)
		agent = Agent(model=model, tools=[ping])

		result = agent.run("hi")

		tool_messages = [m for m in result.messages if m["role"] == "tool"]
		self.assertEqual([m["tool_call_id"] for m in tool_messages], ["c1", "c2"])
		self.assertEqual(len(result.tool_calls), 2)


class TestAgentInputValidation(UnitTestCase):
	def test_non_list_non_str_input_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(TypeError):
			agent.run(42)

	def test_non_dict_element_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(TypeError):
			agent.run(["not a dict"])

	def test_missing_role_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(ValueError) as ctx:
			agent.run([{"content": "hi"}])
		self.assertIn("role", str(ctx.exception))

	def test_invalid_role_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(ValueError):
			agent.run([{"role": "robot", "content": "hi"}])

	def test_tool_message_without_tool_call_id_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(ValueError) as ctx:
			agent.run([{"role": "tool", "content": "result"}])
		self.assertIn("tool_call_id", str(ctx.exception))

	def test_message_without_content_or_tool_calls_rejected(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(ValueError):
			agent.run([{"role": "user"}])

	def test_assistant_message_with_only_tool_calls_accepted(self):
		model = FakeModel([_final("ok")])
		agent = Agent(model=model)
		agent.run(
			[
				{
					"role": "assistant",
					"tool_calls": [
						{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
					],
				},
				{"role": "tool", "tool_call_id": "c1", "content": "result"},
			]
		)


class TestAgentLimits(UnitTestCase):
	def test_max_iterations_raises_when_exceeded(self):
		looping = ChatResponse(
			content=None,
			tool_calls=[ToolCall(id="c", name="ping", arguments={})],
		)

		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel([looping, looping, looping])
		agent = Agent(model=model, tools=[ping], max_iterations=2)

		with self.assertRaises(RuntimeError) as ctx:
			agent.run("hi")
		self.assertIn("max_iterations", str(ctx.exception))

	def _lookup_tool(self):
		@tool
		def lookup(query: str) -> dict:
			"""Look something up."""
			return {"query": query, "page": {"title": "Guide", "url": "https://docs.example/guide", "text": "Step one. Step two."}}

		return lookup

	def _search(self, i):
		return _tool_call("lookup", {"query": f"words {i}"}, call_id=f"c{i}")

	def test_same_tool_every_step_ends_with_the_docs_found(self):
		model = FakeModel([self._search(i) for i in range(1, 9)])
		agent = Agent(model=model, tools=[self._lookup_tool()], max_iterations=8)

		result = agent.run("how do I do this")

		self.assertEqual(result.iterations, 3)
		self.assertEqual(len(model.calls), 3)
		self.assertIn("Step one. Step two.", result.output)
		self.assertIn("https://docs.example/guide", result.output)

	def test_stream_same_tool_every_step_ends_with_the_docs_found(self):
		model = FakeModel([self._search(i) for i in range(1, 9)])
		agent = Agent(model=model, tools=[self._lookup_tool()], max_iterations=8)

		events = list(agent.run("how do I do this", stream=True))

		self.assertEqual(len(model.calls), 3)
		self.assertIsInstance(events[-1], Done)
		self.assertIn("https://docs.example/guide", events[-1].result.output)
		self.assertTrue(any(isinstance(e, TextChunk) and "Step one" in e.text for e in events))

	def test_different_tools_between_steps_do_not_count_as_a_loop(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel(
			[
				self._search(1),
				_tool_call("ping", {}, call_id="p1"),
				self._search(2),
				_tool_call("ping", {}, call_id="p2"),
				_final("done"),
			]
		)
		agent = Agent(model=model, tools=[self._lookup_tool(), ping], max_iterations=8)

		self.assertEqual(agent.run("hi").output, "done")

	def test_running_out_of_steps_returns_the_docs_found_instead_of_raising(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel([self._search(1), _tool_call("ping", {}, call_id="p1"), _tool_call("ping", {}, call_id="p2")])
		agent = Agent(model=model, tools=[self._lookup_tool(), ping], max_iterations=3)

		result = agent.run("how do I do this")

		self.assertIn("https://docs.example/guide", result.output)

	def test_search_results_without_a_docs_page_are_listed(self):
		@tool
		def lookup(query: str) -> dict:
			"""Search."""
			return {"query": query, "results": [{"title": "A page", "url": "https://web.example/a"}]}

		model = FakeModel([_tool_call("lookup", {"query": f"w{i}"}, call_id=f"c{i}") for i in range(1, 6)])
		agent = Agent(model=model, tools=[lookup], max_iterations=5)

		self.assertIn("- A page: https://web.example/a", agent.run("hi").output)

	def test_docs_from_an_earlier_turn_are_not_shown_for_a_new_question(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		earlier = [
			{"role": "user", "content": "first question"},
			{"role": "tool", "tool_call_id": "old", "content": json.dumps({"page": {"url": "https://docs.example/old", "text": "old"}})},
			{"role": "assistant", "content": "first answer"},
			{"role": "user", "content": "second question"},
		]
		model = FakeModel([_tool_call("ping", {}, call_id=f"p{i}") for i in range(1, 4)])
		agent = Agent(model=model, tools=[ping], max_iterations=3)

		with self.assertRaises(RuntimeError):
			agent.run(earlier)

	def test_max_iterations_must_be_positive(self):
		with self.assertRaises(ValueError):
			Agent(model=FakeModel([]), max_iterations=0)

	def test_duplicate_tool_names_rejected(self):
		@tool(name="dup")
		def a() -> str:
			"""A."""
			return "a"

		@tool(name="dup")
		def b() -> str:
			"""B."""
			return "b"

		with self.assertRaises(ValueError):
			Agent(model=FakeModel([]), tools=[a, b])


def _ask(prompt: str, options: list[str] | None = None, multi_select: bool = False) -> Question:
	return Question(prompt=prompt, options=options or [], multi_select=multi_select)


class TestAgentConsultation(UnitTestCase):
	def test_question_returning_tool_pauses(self):
		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask the user."""
			return _ask(prompt, options=["Customers", "Leads"])

		model = FakeModel([_tool_call("ask_user", {"prompt": "Which list?"}, call_id="c1")])
		agent = Agent(model=model, tools=[ask_user])

		result = agent.run("email the list")

		self.assertTrue(result.paused)
		self.assertEqual(len(result.questions), 1)
		question = result.questions[0]
		self.assertEqual(question.prompt, "Which list?")
		self.assertEqual(question.options, ["Customers", "Leads"])
		self.assertEqual(question.key, "c1")
		self.assertFalse([m for m in result.messages if m["role"] == "tool"])

	def test_non_question_tools_still_loop(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel([_tool_call("ping", {}), _final("done")])
		agent = Agent(model=model, tools=[ping])

		result = agent.run("hi")

		self.assertFalse(result.paused)
		self.assertEqual(result.iterations, 2)

	def test_resume_feeds_answer_back_as_result(self):
		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask the user."""
			return _ask(prompt, options=["Customers", "Leads"])

		agent = Agent(
			model=FakeModel([_tool_call("ask_user", {"prompt": "Which list?"}, call_id="c1")]),
			tools=[ask_user],
		)
		paused = agent.run("email the list")

		agent.model = FakeModel([_final("emailed the customers")])
		result = agent.resume(paused.messages, {"c1": "Customers"})

		self.assertFalse(result.paused)
		self.assertEqual(result.output, "emailed the customers")
		tool_msg = next(m for m in result.messages if m["role"] == "tool")
		self.assertEqual(tool_msg["tool_call_id"], "c1")
		self.assertEqual(tool_msg["content"], "Customers")

	def test_resume_serializes_multiselect_answer(self):
		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask the user."""
			return _ask(prompt, options=["A", "B", "C"], multi_select=True)

		agent = Agent(
			model=FakeModel([_tool_call("ask_user", {"prompt": "Which?"}, call_id="c1")]),
			tools=[ask_user],
		)
		paused = agent.run("pick")

		agent.model = FakeModel([_final("ok")])
		result = agent.resume(paused.messages, {"c1": ["A", "C"]})

		tool_msg = next(m for m in result.messages if m["role"] == "tool")
		self.assertEqual(json.loads(tool_msg["content"]), ["A", "C"])

	def test_other_tools_run_while_a_question_pauses(self):
		@tool
		def safe() -> str:
			"""Safe."""
			return "safe-result"

		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask the user."""
			return _ask(prompt)

		response = ChatResponse(
			content=None,
			tool_calls=[
				ToolCall(id="c1", name="safe", arguments={}),
				ToolCall(id="c2", name="ask_user", arguments={"prompt": "ok?"}),
			],
		)
		agent = Agent(model=FakeModel([response]), tools=[safe, ask_user])

		result = agent.run("do both")

		self.assertTrue(result.paused)
		self.assertEqual([c.name for c in result.tool_calls], ["safe"])
		self.assertEqual([q.key for q in result.questions], ["c2"])
		tool_msgs = [m for m in result.messages if m["role"] == "tool"]
		self.assertEqual(len(tool_msgs), 1)
		self.assertEqual(tool_msgs[0]["content"], "safe-result")

	def test_resume_without_pending_raises(self):
		agent = Agent(model=FakeModel([]))
		with self.assertRaises(ValueError):
			agent.resume([{"role": "user", "content": "hi"}], {})


class TestAgentStreaming(UnitTestCase):
	def test_run_stream_yields_text_chunks_then_done(self):
		model = FakeModel([_final("hello world")], streams=[["hello ", "world"]])
		agent = Agent(model=model)

		events = list(agent.run("hi", stream=True))

		texts = [e.text for e in events if isinstance(e, TextChunk)]
		self.assertEqual(texts, ["hello ", "world"])

		done = events[-1]
		self.assertIsInstance(done, Done)
		self.assertIsInstance(done.result, RunResult)
		self.assertEqual(done.result.output, "hello world")
		self.assertEqual(done.result.iterations, 1)
		self.assertFalse(done.result.paused)

	def test_run_stream_passes_stream_flag_to_model(self):
		model = FakeModel([_final("ok")])
		agent = Agent(model=model)

		list(agent.run("hi", stream=True))

		self.assertTrue(model.calls[0]["stream"])

	def test_run_stream_emits_tool_started_and_ended_around_invocation(self):
		@tool
		def add(a: int, b: int) -> int:
			"""Add."""
			return a + b

		model = FakeModel(
			[_tool_call("add", {"a": 2, "b": 3}), _final("five")],
			streams=[[""], ["five"]],
		)
		agent = Agent(model=model, tools=[add])

		events = list(agent.run("add 2 and 3", stream=True))

		started = [e for e in events if isinstance(e, ToolStarted)]
		ended = [e for e in events if isinstance(e, ToolEnded)]
		self.assertEqual(len(started), 1)
		self.assertEqual(started[0].name, "add")
		self.assertEqual(started[0].arguments, {"a": 2, "b": 3})
		self.assertEqual(ended[0].result, "5")

		done = events[-1]
		self.assertIsInstance(done, Done)
		self.assertEqual(done.result.output, "five")
		self.assertEqual(done.result.iterations, 2)

	def test_run_stream_paused_ends_with_done_paused(self):
		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask."""
			return Question(prompt=prompt, options=["A", "B"])

		model = FakeModel([_tool_call("ask_user", {"prompt": "Which?"}, call_id="c1")])
		agent = Agent(model=model, tools=[ask_user])

		events = list(agent.run("pick", stream=True))

		done = events[-1]
		self.assertIsInstance(done, Done)
		self.assertTrue(done.result.paused)
		self.assertEqual(done.result.questions[0].key, "c1")

	def test_run_stream_accumulates_usage_across_iterations(self):
		@tool
		def ping() -> str:
			"""Ping."""
			return "pong"

		model = FakeModel(
			[
				ChatResponse(
					content=None,
					tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
					usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
				),
				_final("done", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
			]
		)
		agent = Agent(model=model, tools=[ping])

		events = list(agent.run("hi", stream=True))

		done = events[-1]
		self.assertEqual(done.result.usage, {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11})

	def test_run_stream_resume_continues_with_answer(self):
		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask."""
			return Question(prompt=prompt, options=["A", "B"])

		agent = Agent(
			model=FakeModel([_tool_call("ask_user", {"prompt": "Which?"}, call_id="c1")]),
			tools=[ask_user],
		)
		paused_events = list(agent.run("pick", stream=True))
		paused = paused_events[-1].result

		agent.model = FakeModel([_final("picked A")], streams=[["picked A"]])
		resumed_events = list(agent.resume(paused.messages, {"c1": "A"}, stream=True))

		texts = [e.text for e in resumed_events if isinstance(e, TextChunk)]
		self.assertEqual("".join(texts), "picked A")
		done = resumed_events[-1]
		self.assertIsInstance(done, Done)
		self.assertEqual(done.result.output, "picked A")

	def test_run_stream_resume_preserves_prior_tool_calls(self):
		@tool
		def lookup() -> str:
			"""Lookup."""
			return "result"

		@tool
		def ask_user(prompt: str) -> Question:
			"""Ask."""
			return Question(prompt=prompt, options=["A", "B"])

		agent = Agent(
			model=FakeModel(
				[
					_tool_call("lookup", {}, call_id="c1"),
					_tool_call("ask_user", {"prompt": "Which?"}, call_id="c2"),
				]
			),
			tools=[lookup, ask_user],
		)
		paused = list(agent.run("look then ask", stream=True))[-1].result
		self.assertEqual([c.name for c in paused.tool_calls], ["lookup"])

		agent.model = FakeModel([_final("done")], streams=[["done"]])
		done = list(agent.resume(paused.messages, {"c2": "A"}, stream=True))[-1]

		self.assertEqual([c.name for c in done.result.tool_calls], ["lookup", "ask_user"])


class TestAgentConfirmation(UnitTestCase):
	def _danger_tool(self):
		calls: list[dict[str, Any]] = []

		@tool(requires_confirmation=True)
		def write_file(path: str, body: str) -> str:
			"""Write text to a file."""
			calls.append({"path": path, "body": body})
			return f"wrote {len(body)} bytes to {path}"

		return write_file, calls

	def test_confirmation_required_tool_pauses_run_without_executing(self):
		write_file, calls = self._danger_tool()
		model = FakeModel([_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1")])
		agent = Agent(model=model, tools=[write_file])

		result = agent.run("write hi to /tmp/x")

		self.assertTrue(result.paused)
		self.assertEqual(len(result.questions), 1)
		self.assertEqual(result.questions[0].options, ["Approve", "Deny"])
		self.assertTrue(result.questions[0].allow_other)
		self.assertIn("write_file", result.questions[0].prompt)
		self.assertEqual(calls, [])  # tool never ran

	def test_auto_approve_runs_confirmation_tool_without_pausing(self):
		write_file, calls = self._danger_tool()
		model = FakeModel(
			[_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1"), _final("done")]
		)
		agent = Agent(model=model, tools=[write_file], auto_approve=True)

		result = agent.run("write hi to /tmp/x")

		self.assertFalse(result.paused)
		self.assertEqual(calls, [{"path": "/tmp/x", "body": "hi"}])  # ran unattended
		self.assertEqual(result.output, "done")

	def test_approve_invokes_tool_and_feeds_real_result_to_llm(self):
		write_file, calls = self._danger_tool()
		pause_response = _tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1")
		final_response = _final("done")
		model = FakeModel([pause_response, final_response])
		agent = Agent(model=model, tools=[write_file])

		paused = agent.run("write hi to /tmp/x")
		resumed = agent.resume(paused.messages, {"c1": "Approve"})

		self.assertEqual(calls, [{"path": "/tmp/x", "body": "hi"}])
		tool_message = next(m for m in resumed.messages if m["role"] == "tool")
		self.assertEqual(tool_message["content"], "wrote 2 bytes to /tmp/x")
		self.assertEqual(resumed.output, "done")

	def test_deny_records_rejection_and_stops_without_calling_model(self):
		write_file, calls = self._danger_tool()
		# Only the pausing response is scripted: if resume called the model again,
		# FakeModel would raise "ran out of scripted responses".
		model = FakeModel([_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1")])
		agent = Agent(model=model, tools=[write_file])

		paused = agent.run("write hi to /tmp/x")
		resumed = agent.resume(paused.messages, {"c1": "Deny"})

		self.assertEqual(calls, [])  # tool never ran
		self.assertFalse(resumed.paused)  # run ended, not paused
		self.assertIsNone(resumed.output)
		self.assertEqual(len(model.calls), 1)  # model not called again after Deny
		tool_message = next(m for m in resumed.messages if m["role"] == "tool")
		self.assertEqual(
			json.loads(tool_message["content"]),
			{"status": "denied", "message": "User denied this tool call."},
		)

	def test_stream_deny_stops_and_replays_denied_call(self):
		write_file, calls = self._danger_tool()
		model = FakeModel([_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1")])
		agent = Agent(model=model, tools=[write_file])

		paused = agent.run("write hi to /tmp/x")
		events = list(agent.resume(paused.messages, {"c1": "Deny"}, stream=True))

		self.assertEqual(calls, [])  # tool never ran
		self.assertEqual(len(model.calls), 1)  # model not called again after Deny
		# the denied call is replayed as a ToolEnded so the UI can fill its card, then Done
		tool_ended = [e for e in events if isinstance(e, ToolEnded)]
		self.assertEqual(len(tool_ended), 1)
		self.assertEqual(json.loads(tool_ended[0].result)["status"], "denied")
		self.assertIsInstance(events[-1], Done)
		self.assertFalse(events[-1].result.paused)

	def test_confirm_prompt_renders_plain_english_body(self):
		@tool(
			requires_confirmation=True,
			confirm_prompt=lambda args: f"Send email to {args['to']}",
		)
		def send_email(to: str, body: str) -> str:
			"""Send an email."""
			return "sent"

		model = FakeModel([_tool_call("send_email", {"to": "alice@example.com", "body": "hi"}, call_id="c1")])
		agent = Agent(model=model, tools=[send_email])

		result = agent.run("email alice")

		self.assertIn("Send email to alice@example.com", result.questions[0].prompt)
		self.assertNotIn("body", result.questions[0].prompt)  # raw JSON shape isn't leaking through

	def test_free_text_answer_redirects_to_llm_with_feedback(self):
		write_file, calls = self._danger_tool()
		model = FakeModel(
			[
				_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c1"),
				_final("ok"),
			]
		)
		agent = Agent(model=model, tools=[write_file])

		paused = agent.run("write hi")
		resumed = agent.resume(paused.messages, {"c1": "use /tmp/y instead"})

		self.assertEqual(calls, [])  # tool never ran
		tool_message = next(m for m in resumed.messages if m["role"] == "tool")
		payload = json.loads(tool_message["content"])
		self.assertEqual(payload["status"], "redirect")
		self.assertEqual(payload["user_feedback"], "use /tmp/y instead")

	def test_resume_preserves_tool_calls_from_before_the_pause(self):
		write_file, calls = self._danger_tool()

		@tool
		def read_file(path: str) -> str:
			"""Read a file."""
			return "contents"

		model = FakeModel(
			[
				_tool_call("read_file", {"path": "/tmp/a"}, call_id="c1"),
				_tool_call("write_file", {"path": "/tmp/x", "body": "hi"}, call_id="c2"),
			]
		)
		agent = Agent(model=model, tools=[read_file, write_file])

		paused = agent.run("read then write")
		self.assertEqual([c.name for c in paused.tool_calls], ["read_file"])

		agent.model = FakeModel([_final("done")])
		resumed = agent.resume(paused.messages, {"c2": "Approve"})

		self.assertEqual(calls, [{"path": "/tmp/x", "body": "hi"}])  # confirmation ran
		# Both the pre-pause call and the just-approved call survive on the resumed result.
		self.assertEqual([c.name for c in resumed.tool_calls], ["read_file", "write_file"])


class TestQuestion(UnitTestCase):
	def test_defaults_are_open_text_with_other(self):
		q = Question(prompt="What subject line?")
		self.assertEqual(q.options, [])
		self.assertFalse(q.multi_select)
		self.assertTrue(q.allow_other)

	def test_holds_llm_authored_options(self):
		q = Question(
			prompt="I'll delete 5 inactive customers.",
			options=["Go ahead", "Only the 2 oldest"],
			multi_select=False,
			key="c1",
		)
		self.assertEqual(q.options, ["Go ahead", "Only the 2 oldest"])
		self.assertTrue(q.allow_other)
		self.assertEqual(q.key, "c1")


class TestToolCallParsing(UnitTestCase):
	def test_invalid_json_becomes_error_not_raise(self):
		from flow.lib.model import _build_tool_call

		call = _build_tool_call("c1", "read", "{not json")
		self.assertEqual(call.arguments, {})
		self.assertIn("Invalid JSON", call.error)

	def test_non_object_json_is_error(self):
		from flow.lib.model import _build_tool_call

		call = _build_tool_call("c1", "read", "[1, 2]")
		self.assertIn("must be a JSON object", call.error)

	def test_valid_json_parses_without_error(self):
		from flow.lib.model import _build_tool_call

		call = _build_tool_call("c1", "read", '{"doctype": "User"}')
		self.assertIsNone(call.error)
		self.assertEqual(call.arguments, {"doctype": "User"})


class TestAgentMalformedToolCall(UnitTestCase):
	def test_parse_error_feeds_back_and_run_continues(self):
		bad = ChatResponse(
			content=None,
			tool_calls=[
				ToolCall(id="c1", name="read", arguments={}, error="Invalid JSON in arguments for 'read'.")
			],
		)
		model = FakeModel([bad, _final("recovered")])
		agent = Agent(model=model)

		result = agent.run("read something")

		self.assertFalse(result.paused)
		self.assertEqual(result.output, "recovered")
		tool_msg = next(m for m in result.messages if m["role"] == "tool")
		self.assertIn("Invalid JSON", json.loads(tool_msg["content"])["error"])


class TestBrowserModelContext(UnitTestCase):
	"""A browser model's prompt carries only relevant tools and recent turns."""

	def _schemas(self, *names):
		return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in names]

	def test_tools_follow_the_request(self):
		from flow.lib.agent import _relevant_tools

		schemas = self._schemas(
			"show_records", "count", "read", "find_doctypes", "create", "required_values",
			"error_diagnosis", "document_flow", "assignment_steps", "status_steps",
			"role_permissions", "explain_doctype", "creation_steps", "small_talk",
		)
		pick = lambda text: [t["function"]["name"] for t in _relevant_tools([{"role": "user", "content": text}], schemas)]  # noqa: E731
		self.assertEqual(pick("show me the invoices"), ["show_records", "count", "read", "find_doctypes"])
		self.assertEqual(pick("create a quotation")[:2], ["create", "required_values"])
		self.assertEqual(pick("why does my invoice fail"), ["error_diagnosis"])
		# Words like "issue" or "problem" name a field or a topic as often as a failure.
		self.assertNotIn("error_diagnosis", pick("what is the issue date of the invoice"))
		self.assertEqual(pick("I can't submit the invoice"), ["error_diagnosis"])

		# A how-to with no dedicated tool (deleting, renaming...) goes to the docs, not to a
		# tool that explains what comes after a document.
		with_docs = self._schemas("document_flow", "explain_doctype", "docs_howto", "frappe_docs", "web_search")
		names = [t["function"]["name"] for t in _relevant_tools([{"role": "user", "content": "how to delete the user"}], with_docs)]
		self.assertEqual(names, ["docs_howto", "frappe_docs", "web_search"])
		self.assertEqual(pick("how to assign roles to users"), ["assignment_steps", "explain_doctype"])
		self.assertEqual(pick("how do I mark this invoice as paid"), ["status_steps", "document_flow"])
		self.assertEqual(pick("what permissions does Sales User have"), ["role_permissions", "explain_doctype"])
		self.assertEqual(pick("how many customers exist"), ["count", "show_records"])
		# small_talk is never offered to the model
		self.assertNotIn("small_talk", pick("hello there"))

	def test_tools_used_this_turn_stay_available(self):
		from flow.lib.agent import _relevant_tools

		schemas = self._schemas("show_records", "count", "read", "find_doctypes", "describe")
		messages = [
			{"role": "user", "content": "show me the invoices"},
			{"role": "assistant", "content": None, "tool_calls": [{"id": "a", "function": {"name": "describe", "arguments": "{}"}}]},
		]
		self.assertEqual(_relevant_tools(messages, schemas)[0]["function"]["name"], "describe")

	def test_only_recent_turns_are_sent(self):
		from flow.lib.agent import RECENT_USER_TURNS, _recent_turns

		messages = [{"role": "system", "content": "rules"}]
		for i in range(5):
			messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
		trimmed = _recent_turns(messages)
		self.assertEqual(trimmed[0]["role"], "system")
		self.assertEqual([m["content"] for m in trimmed if m["role"] == "user"], [f"q{i}" for i in range(5 - RECENT_USER_TURNS, 5)])


class TestReferencedDocType(UnitTestCase):
	""""this <word>" in a follow-up is read as the DocType of the record shown before."""

	def _history(self, doctype):
		import json

		return [
			{"role": "user", "content": "show my open todos"},
			{"role": "tool", "tool_call_id": "c1", "content": json.dumps({"answer": "...", "doctype": doctype})},
		]

	def test_this_word_becomes_the_shown_doctype(self):
		from flow.lib.agent import _with_referenced_doctype

		msgs = self._history("Sales Invoice") + [{"role": "user", "content": "how can I make this invoice paid?"}]
		self.assertEqual(_with_referenced_doctype(msgs[-1]["content"], msgs), "how can I make this Sales Invoice paid?")

	def test_bare_this_or_it_gets_the_shown_doctype(self):
		from flow.lib.agent import _with_referenced_doctype

		for text, expected in (
			("how can I make this as paid?", "how can I make this Sales Invoice as paid?"),
			("how do I mark this paid", "how do I mark this Sales Invoice paid"),
			("how can I make it paid?", "how can I make the Sales Invoice paid?"),
		):
			msgs = self._history("Sales Invoice") + [{"role": "user", "content": text}]
			self.assertEqual(_with_referenced_doctype(text, msgs), expected)

	def test_unrelated_word_is_left_alone(self):
		from flow.lib.agent import _with_referenced_doctype

		msgs = self._history("Sales Invoice") + [{"role": "user", "content": "how do I cancel that order?"}]
		self.assertEqual(_with_referenced_doctype(msgs[-1]["content"], msgs), "how do I cancel that order?")


class TestAccessFollowUp(IntegrationTestCase):
	"""A who-has-access question that names no record type means the one talked about last."""

	def _history(self, doctype):
		import json

		return [
			{"role": "user", "content": "who can create a sales invoice?"},
			{"role": "tool", "tool_call_id": "c1", "content": json.dumps({"answer": "...", "doctype": doctype})},
		]

	def test_the_last_doctype_is_spelled_out(self):
		from flow.lib.agent import _with_referenced_doctype

		text = "which user has the access in this system?"
		msgs = self._history("Sales Invoice") + [{"role": "user", "content": text}]
		self.assertEqual(_with_referenced_doctype(text, msgs), "which user has the access in this system for Sales Invoice")

	def test_the_open_page_comes_first(self):
		from flow.lib.agent import _with_referenced_doctype

		text = "who has access?"
		msgs = self._history("Sales Invoice") + [{"role": "user", "content": text}]
		self.assertEqual(_with_referenced_doctype(text, msgs, {"doctype": "Item", "name": ""}), "who has access for Item")

	def test_a_message_that_names_a_doctype_or_is_not_about_access_is_left_alone(self):
		from flow.lib.agent import _with_referenced_doctype

		for text in ("who can create an item?", "show all customers", "how many users are there"):
			msgs = self._history("Sales Invoice") + [{"role": "user", "content": text}]
			self.assertEqual(_with_referenced_doctype(text, msgs), text)

	def test_with_nothing_talked_about_yet_it_is_left_alone(self):
		from flow.lib.agent import _with_referenced_doctype

		text = "which user has the access in this system?"
		self.assertEqual(_with_referenced_doctype(text, [{"role": "user", "content": text}]), text)

	def test_the_follow_up_routes_to_doctype_access_without_a_model(self):
		from flow.tools.answers import doctype_access

		text = "which user has the access in this system?"
		agent = Agent(model=FakeModel([]), tools=[doctype_access])
		response = agent._routed_response(self._history("ToDo") + [{"role": "user", "content": text}])
		self.assertEqual([(c.name, c.arguments) for c in response.tool_calls], [("doctype_access", {"doctype": "ToDo"})])


class TestPageContext(IntegrationTestCase):
	"""The desk page the user is on resolves "this" / "it" and is told to the model."""

	def setUp(self):
		self.todo = frappe.get_doc({"doctype": "ToDo", "description": "page context probe"}).insert()
		self.page = {"doctype": "ToDo", "name": self.todo.name}

	def test_this_means_the_open_record(self):
		from flow.lib.agent import _with_referenced_doctype

		msgs = [{"role": "user", "content": "what is the description of this todo?"}]
		self.assertEqual(
			_with_referenced_doctype(msgs[0]["content"], msgs, self.page),
			f"what is the description of this ToDo {self.todo.name}?",
		)
		msgs = [{"role": "user", "content": "how do I close it?"}]
		self.assertEqual(_with_referenced_doctype(msgs[0]["content"], msgs, self.page), f"how do I close the ToDo {self.todo.name}?")

	def test_open_record_routes_its_question(self):
		agent = frappe.get_doc("Flow Agent", "Flow Lite").assemble()
		agent.page = self.page
		call = agent._routed_response([{"role": "user", "content": "what is the description of this todo?"}]).tool_calls[0]
		self.assertEqual((call.name, call.arguments["names"], call.arguments["about"]), ("show_records", [self.todo.name], ["description"]))

	def test_model_is_told_the_page(self):
		from flow.lib.agent import _with_page_note

		out = _with_page_note([{"role": "system", "content": "RULES"}, {"role": "user", "content": "hi"}], self.page)
		self.assertEqual(len(out), 2)
		self.assertIn(f"ToDo {self.todo.name} open", out[0]["content"])

	def test_page_must_be_readable_and_real(self):
		from flow.api.api import _page

		self.assertEqual(_page("ToDo", self.todo.name), self.page)
		self.assertEqual(_page("ToDo", "no-such-todo"), {"doctype": "ToDo"})
		self.assertIsNone(_page("No Such DocType", "x"))

	def test_open_record_question_without_fields_goes_to_the_model(self):
		agent = frappe.get_doc("Flow Agent", "Flow Lite").assemble()
		agent.page = self.page
		frappe.conf.flow_laya_enabled = 0
		try:
			self.assertIsNone(agent._routed_response([{"role": "user", "content": "why is this so late?"}]))
		finally:
			frappe.conf.flow_laya_enabled = None



class TestGuessedArguments(IntegrationTestCase):
	"""A model-chosen call about a record type nobody mentioned is bounced back, not answered."""

	def _invoke(self, call_id, doctype, user_text, page=None, tool_text=None):
		import json

		from flow.lib.agent import Agent
		from flow.lib.model import ToolCall
		from flow.tools.answers import explain_doctype

		messages = [{"role": "user", "content": user_text}]
		if tool_text:
			messages.append({"role": "tool", "tool_call_id": "t1", "content": tool_text})
		agent = Agent(model=FakeModel([]), tools=[explain_doctype])
		agent.page = page
		raw = agent._invoke(ToolCall(id=call_id, name="explain_doctype", arguments={"doctype": doctype}), messages)
		return json.loads(raw) if isinstance(raw, str) else raw

	def test_an_unmentioned_doctype_is_refused_so_the_model_asks(self):
		result = self._invoke("call_1", "ToDo", "what is this")
		self.assertIn("Nobody mentioned", result["error"])

	def test_a_doctype_the_user_named_in_any_form_is_answered(self):
		for text in ("what is a todo", "explain todos please"):
			self.assertIn("answer", self._invoke("call_1", "ToDo", text))

	def test_the_open_page_or_an_earlier_result_counts_as_mentioned(self):
		self.assertIn("answer", self._invoke("call_1", "ToDo", "what is this", page={"doctype": "ToDo", "name": ""}))
		self.assertIn("answer", self._invoke("call_1", "ToDo", "what is this", tool_text='{"doctype": "ToDo"}'))

	def test_a_routed_call_is_not_second_guessed(self):
		# A route reads the DocType from the user's own message (even misspelt), so it is trusted.
		self.assertIn("answer", self._invoke("route_abc", "ToDo", "what is this"))


class TestScopeCheck(IntegrationTestCase):
	"""A browser model is asked one IN / OUT question before it answers a message no route took."""

	def _run(self, replies, text="write a short poem about rain"):
		from flow.lib.agent import Agent, Done, TextChunk

		model = FakeModel([ChatResponse(content=r) for r in replies])
		model.model_id = "webllm/test-model"
		events = list(Agent(model=model, tools=[]).run(text, stream=True))
		shown = "".join(e.text for e in events if isinstance(e, TextChunk))
		output = next(e.result.output for e in events if isinstance(e, Done))
		return model, shown, output

	def test_out_ends_the_turn_with_the_fixed_reply_and_one_model_call(self):
		from flow.tools.web import OUT_OF_SCOPE

		model, shown, output = self._run(["OUT"])
		self.assertEqual((shown, output), (OUT_OF_SCOPE, OUT_OF_SCOPE))
		self.assertEqual(len(model.calls), 1)

	def test_in_goes_on_to_the_answer_and_the_word_is_never_shown(self):
		model, shown, output = self._run(["IN", "Rain is water."], text="what is rain on the site")
		self.assertEqual((shown, output), ("Rain is water.", "Rain is water."))
		self.assertEqual(len(model.calls), 2)

	def test_a_record_type_of_the_site_is_pointed_out_to_the_model(self):
		from unittest.mock import patch

		# With no site vocabulary to skip the check, the record type is still pointed out to the model.
		with patch("flow.lib.agent._site_vocabulary", return_value=[]):
			model, _shown, _output = self._run(["IN", "Open the Customer list."], text="where can I find the customer?")
		asked = str(model.calls[0])
		self.assertIn("record type named Customer", asked)

	def test_two_record_types_in_a_message_skip_the_check(self):
		model, _shown, output = self._run(["An answer."], text="is it possible to assign a todo to a note?")
		self.assertEqual((output, len(model.calls)), ("An answer.", 1))  # no IN / OUT question asked

	def test_a_word_from_the_sites_business_vocabulary_skips_the_check(self):
		from unittest.mock import patch

		with patch("flow.lib.agent._site_vocabulary", return_value=["leave", "stock"]):
			model, _shown, output = self._run(["An answer."], text="who is on leave today?")
			self.assertEqual((output, len(model.calls)), ("An answer.", 1))  # no IN / OUT question
			model, _shown, output = self._run(["OUT"], text="write a short poem about rain")
			self.assertEqual(len(model.calls), 1)  # asked, and refused

	def test_the_vocabulary_comes_from_the_names_of_business_record_types(self):
		from flow.lib.agent import _site_vocabulary, _uses_site_words

		words = _site_vocabulary()
		self.assertTrue(words)
		self.assertTrue(all(w.isalpha() and len(w) >= 4 for w in words))
		self.assertTrue(_uses_site_words(f"tell me about {words[0]}s please"))
		self.assertFalse(_uses_site_words("zzqx wyvv"))

	def test_a_follow_up_is_checked_beside_the_earlier_exchange(self):
		from flow.lib.agent import Agent
		from flow.lib.agent import TextChunk  # noqa: F401

		model = FakeModel([ChatResponse(content="IN"), ChatResponse(content="An answer.")])
		model.model_id = "webllm/test-model"
		messages = [
			{"role": "user", "content": "a question about the site"},
			{"role": "assistant", "content": "Which site do you mean?"},
			{"role": "user", "content": "In this site"},
		]
		list(Agent(model=model, tools=[]).run(messages, stream=True))
		asked = str(model.calls[0])
		self.assertIn("Earlier reply: Which site do you mean?", asked)
		self.assertIn("Earlier message: a question about the site", asked)

	def test_no_hint_when_the_message_names_no_record_type(self):
		model, _shown, _output = self._run(["OUT"], text="write a short poem about rain")
		self.assertNotIn("record type named", str(model.calls[0]))

	def test_an_unclear_reply_counts_as_in(self):
		model, _shown, output = self._run(["Hmm, maybe?", "An answer."])
		self.assertEqual(output, "An answer.")

	def test_server_models_and_a_site_switch_skip_the_check(self):
		from flow.lib.agent import Agent, Done

		plain = FakeModel([ChatResponse(content="Direct.")])
		events = list(Agent(model=plain, tools=[]).run("write a poem", stream=True))
		self.assertEqual(next(e.result.output for e in events if isinstance(e, Done)), "Direct.")
		frappe.conf.flow_scope_check = 0
		self.addCleanup(lambda: frappe.conf.pop("flow_scope_check", None))
		model, _shown, output = self._run(["Direct."])
		self.assertEqual((output, len(model.calls)), ("Direct.", 1))


class TestDocsPick(IntegrationTestCase):
	"""The browser model chooses among the docs sections a how-to lookup found."""

	OPTIONS = [{"label": "A: first", "answer": "First section.\n\nSource: u1"}, {"label": "B: second", "answer": "Second section.\n\nSource: u2"}]

	def _run(self, reply, model_id="webllm/test-model"):
		from flow.lib.agent import Agent

		model = FakeModel([ChatResponse(content=reply)])
		model.model_id = model_id
		agent = Agent(model=model, tools=[])
		events = []
		gen = agent._pick_option([{"role": "user", "content": "how do I do the thing"}], {"answer": "First section.\n\nSource: u1", "options": list(self.OPTIONS)})
		try:
			while True:
				events.append(next(gen))
		except StopIteration as stop:
			return model, stop.value, events

	def test_the_chosen_section_is_the_answer(self):
		model, result, _events = self._run("2")
		self.assertEqual(result["answer"], "Second section.\n\nSource: u2")
		self.assertNotIn("options", result)
		self.assertEqual(len(model.calls), 1)

	def test_zero_says_the_docs_have_nothing(self):
		from flow.tools.web import NOT_IN_DOCS

		self.assertEqual(self._run("0")[1]["answer"], NOT_IN_DOCS)

	def test_an_unclear_reply_keeps_the_best_match(self):
		for reply in ("hmm", "7", ""):
			self.assertEqual(self._run(reply)[1]["answer"], "First section.\n\nSource: u1", reply)

	def test_server_models_are_not_asked(self):
		model, result, _events = self._run("2", model_id="gpt-test")
		self.assertEqual((result["answer"], len(model.calls)), ("First section.\n\nSource: u1", 0))


class TestApprovedWritesAreCommitted(IntegrationTestCase):
	"""A write the user approved is committed before the model replies, not when the stream ends."""

	def _agent(self):
		from flow.lib.agent import Agent
		from flow.lib.tool import tool

		@tool(requires_confirmation=True)
		def save_thing(fail: bool = False) -> dict:
			"""Save a thing."""
			if fail:
				raise ValueError("nope")
			return {"saved": True}

		@tool
		def read_thing() -> dict:
			"""Read a thing."""
			return {"ok": True}

		return Agent(model=FakeModel([]), tools=[save_thing, read_thing])

	def _commits(self, name, arguments):
		from unittest.mock import patch

		from flow.lib.agent import ToolCall

		agent = self._agent()
		frappe.flags.in_test = False  # the commit is skipped under test; patched out below
		try:
			with patch("frappe.db.commit") as commit:
				agent._run_tool(ToolCall(id="c1", name=name, arguments=arguments))
		finally:
			frappe.flags.in_test = True
		return commit.call_count

	def test_a_write_tool_commits_once_it_has_saved(self):
		self.assertEqual(self._commits("save_thing", {}), 1)

	def test_a_failed_write_commits_nothing(self):
		self.assertEqual(self._commits("save_thing", {"fail": True}), 0)

	def test_a_read_commits_nothing(self):
		self.assertEqual(self._commits("read_thing", {}), 0)


class TestSayWhatFailed(IntegrationTestCase):
	"""A model that ends its turn with no text after a tool failed gives the user the tool's error."""

	def _messages(self, tool_content):
		return [
			{"role": "user", "content": "do it"},
			{"role": "assistant", "content": "", "tool_calls": []},
			{"role": "tool", "tool_call_id": "c1", "content": tool_content},
			{"role": "assistant", "content": ""},
		]

	def test_the_error_is_shown_when_the_model_says_nothing(self):
		from flow.lib.agent import _say_what_failed

		response = ChatResponse(content="")
		messages = self._messages('{"error": "No field called x."}')
		said = _say_what_failed(response, messages)
		self.assertEqual(said, "That did not work: No field called x.")
		self.assertEqual(response.content, said)

	def test_a_model_that_replied_is_left_alone(self):
		from flow.lib.agent import _say_what_failed

		self.assertIsNone(_say_what_failed(ChatResponse(content="Done."), self._messages('{"error": "x"}')))

	def test_a_tool_result_without_an_error_is_left_alone(self):
		from flow.lib.agent import _say_what_failed

		self.assertIsNone(_say_what_failed(ChatResponse(content=""), self._messages('{"created": ["a"]}')))
		self.assertIsNone(_say_what_failed(ChatResponse(content=""), self._messages("not json")))
