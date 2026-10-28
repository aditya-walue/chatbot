# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import json
from unittest.mock import patch

import frappe
from frappe.tests import IntegrationTestCase, UnitTestCase

from flow.lib import webllm
from flow.lib.model import Model
from flow.lib.webllm import BrowserRequest, parse_reply, to_plain_messages

TOOL = {
	"type": "function",
	"function": {
		"name": "get_doc",
		"description": "Fetch a document.",
		"parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
	},
}


class TestPlainMessages(UnitTestCase):
	def test_tools_go_into_system_prompt(self):
		out = to_plain_messages(
			[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], [TOOL]
		)
		self.assertEqual([m["role"] for m in out], ["system", "user"])
		self.assertTrue(out[0]["content"].startswith("Be brief."))
		self.assertIn('"name": "get_doc"', out[0]["content"])
		self.assertIn("<tool_call>", out[0]["content"])

	def test_no_system_message_without_instructions_or_tools(self):
		out = to_plain_messages([{"role": "user", "content": "hi"}], None)
		self.assertEqual(out, [{"role": "user", "content": "hi"}])

	def test_tool_calls_and_results_render_as_tagged_text(self):
		out = to_plain_messages(
			[
				{"role": "user", "content": "open ToDo 1"},
				{
					"role": "assistant",
					"content": None,
					"tool_calls": [
						{
							"id": "c1",
							"type": "function",
							"function": {"name": "get_doc", "arguments": json.dumps({"name": "1"})},
						}
					],
				},
				{"role": "tool", "tool_call_id": "c1", "content": '{"status": "Open"}'},
			],
			None,
		)
		self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
		self.assertEqual(
			out[1]["content"],
			'<tool_call>\n{"name": "get_doc", "arguments": {"name": "1"}}\n</tool_call>',
		)
		self.assertEqual(out[2]["content"], '<tool_response>\n{"status": "Open"}\n</tool_response>')

	def test_consecutive_same_role_messages_merge(self):
		out = to_plain_messages(
			[
				{"role": "tool", "tool_call_id": "a", "content": "1"},
				{"role": "tool", "tool_call_id": "b", "content": "2"},
			],
			None,
		)
		self.assertEqual(len(out), 1)
		self.assertEqual(out[0]["content"].count("<tool_response>"), 2)


class TestMalformedCallRendering(UnitTestCase):
	def test_malformed_call_is_shown_as_a_note_not_a_tool_example(self):
		out = to_plain_messages(
			[
				{"role": "user", "content": "create a todo"},
				{"role": "assistant", "content": None, "tool_calls": [
					{"id": "x1", "type": "function", "function": {"name": "invalid_tool_call", "arguments": "{}"}}
				]},
				{"role": "tool", "tool_call_id": "x1", "content": '{"error": "The <tool_call> block was empty."}'},
			],
			None,
		)
		text = json.dumps(out)
		self.assertNotIn("invalid_tool_call", text)
		self.assertIn("(Your last tool call could not be read: The <tool_call> block was empty.)", out[-1]["content"])

	def test_empty_block_says_so(self):
		self.assertIn("was empty", parse_reply("<tool_call>\n", {}).tool_calls[0].error)


class TestParseReply(UnitTestCase):
	def test_plain_text(self):
		response = parse_reply("Hello!", {"total_tokens": 3})
		self.assertEqual(response.content, "Hello!")
		self.assertEqual(response.tool_calls, [])
		self.assertEqual(response.finish_reason, "stop")
		self.assertEqual(response.usage["total_tokens"], 3)

	def test_tool_calls(self):
		response = parse_reply(
			'Let me check.\n<tool_call>\n{"name": "get_doc", "arguments": {"name": "1"}}\n</tool_call>'
			'<tool_call>{"name": "get_doc", "arguments": "{\\"name\\": \\"2\\"}"}</tool_call>',
			{},
		)
		self.assertEqual(response.content, "Let me check.")
		self.assertEqual([c.arguments for c in response.tool_calls], [{"name": "1"}, {"name": "2"}])
		self.assertTrue(all(c.id.startswith("call_") for c in response.tool_calls))
		self.assertEqual(response.finish_reason, "tool_calls")

	def test_unterminated_tool_call_still_parses(self):
		response = parse_reply('<tool_call>{"name": "get_doc", "arguments": {}}', {})
		self.assertIsNone(response.content)
		self.assertEqual(response.tool_calls[0].name, "get_doc")

	def test_missing_list_closer_is_repaired(self):
		# Real Qwen 2.5 3B output: the `records` list is never closed.
		response = parse_reply(
			'<tool_call>{"name": "create", "arguments": {"doctype": "Sales Invoice", "records": '
			'[{"customer": "Acme", "items": [{"description": "ITEM-0001", "rate": 100, "qty": 3}]}}}'
			"</tool_call>",
			{},
		)
		call = response.tool_calls[0]
		self.assertIsNone(call.error)
		self.assertEqual(call.name, "create")
		self.assertEqual(call.arguments["records"][0]["items"][0]["qty"], 3)

	def test_unclosed_brackets_at_end_are_repaired(self):
		response = parse_reply('<tool_call>{"name": "get_doc", "arguments": {"name": "a}]\\"b"', {})
		self.assertEqual(response.tool_calls[0].arguments, {"name": 'a}]"b'})

	def test_unquoted_key_is_repaired(self):
		# Real Qwen 2.5 3B output.
		response = parse_reply(
			'<tool_call>{"name": "describe", arguments: {"doctype": "Sales Invoice"}}</tool_call>', {}
		)
		self.assertIsNone(response.tool_calls[0].error)
		self.assertEqual(response.tool_calls[0].arguments, {"doctype": "Sales Invoice"})

	def test_trailing_comma_is_dropped_but_strings_are_untouched(self):
		response = parse_reply(
			'<tool_call>{"name": "get_doc", "arguments": {"name": "a, }b: c",},}</tool_call>', {}
		)
		self.assertEqual(response.tool_calls[0].arguments, {"name": "a, }b: c"})

	def test_trailing_junk_and_invented_response_are_ignored(self):
		# Real Qwen 2.5 3B output: no closing tag, garbage bytes, then a made-up tool response.
		response = parse_reply(
			'<tool_call>{"name": "count", "arguments": {"doctype": "Employee"}}\ufffd\ufffd\n<tool_call>\n'
			'(tool_response>\n{"doctype": "Employee", "count": 0}\n)</tool_response>',
			{},
		)
		self.assertEqual(response.tool_calls[0].name, "count")
		self.assertEqual(response.tool_calls[0].arguments, {"doctype": "Employee"})
		self.assertIsNone(response.tool_calls[0].error)

	def test_malformed_tool_call_becomes_retryable_error(self):
		response = parse_reply("<tool_call>{not json</tool_call>", {})
		self.assertEqual(len(response.tool_calls), 1)
		self.assertIsNotNone(response.tool_calls[0].error)


class TestBrowserRoundTrip(IntegrationTestCase):
	def test_reply_resumes_the_waiting_stream(self):
		model = Model(model_id="webllm/Qwen2.5-7B-Instruct-q4f16_1-MLC")
		stream = model.chat("hi", stream=True)

		request = next(stream)
		self.assertIsInstance(request, BrowserRequest)
		self.assertEqual(request.model, "Qwen2.5-7B-Instruct-q4f16_1-MLC")
		self.assertEqual(request.messages, [{"role": "user", "content": "hi"}])

		webllm.submit_reply(request.id, {"content": "Hello!", "usage": {}})
		with self.assertRaises(StopIteration) as done:
			next(stream)
		self.assertEqual(done.exception.value.content, "Hello!")

	def test_browser_error_fails_the_call(self):
		stream = Model(model_id="webllm/some-model").chat("hi", stream=True)
		request = next(stream)
		webllm.submit_reply(request.id, {"error": "No WebGPU"})
		with self.assertRaisesRegex(RuntimeError, "No WebGPU"):
			next(stream)

	def test_late_reply_to_finished_request_is_ignored_quietly(self):
		self.assertFalse(webllm.submit_reply("no-such-request", {"content": "late"}))

	def test_other_user_cannot_answer(self):
		stream = Model(model_id="webllm/some-model").chat("hi", stream=True)
		request = next(stream)
		with self.set_user("Guest"), self.assertRaises(frappe.PermissionError):
			webllm.submit_reply(request.id, {"content": "spoofed"})
		stream.close()

	def test_timeout(self):
		stream = Model(model_id="webllm/some-model").chat("hi", stream=True)
		next(stream)
		with patch.object(webllm, "REPLY_TIMEOUT", 0):
			with self.assertRaises(TimeoutError):
				next(stream)

	def test_non_streaming_call_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "browser"):
			Model(model_id="webllm/some-model").chat("hi")


class TestStrictJson(UnitTestCase):
	def test_schema_offers_reply_or_one_of_the_tools_with_its_own_arguments(self):
		schema = webllm.response_schema([TOOL])
		self.assertEqual(schema["anyOf"][0]["required"], ["reply"])
		tool_option = schema["anyOf"][1]
		self.assertEqual(tool_option["properties"]["tool"]["enum"], ["get_doc"])
		self.assertEqual(tool_option["properties"]["arguments"], TOOL["function"]["parameters"])

	def test_strict_replies_parse(self):
		call = parse_reply('{"tool": "get_doc", "arguments": {"name": "1"}}', {}).tool_calls[0]
		self.assertEqual((call.name, call.arguments), ("get_doc", {"name": "1"}))
		answer = parse_reply('{"reply": "There are 3."}', {})
		self.assertEqual((answer.content, answer.tool_calls), ("There are 3.", []))
		# The tagged format still parses (older chats, or the browser's fallback).
		self.assertEqual(parse_reply('<tool_call>{"name": "get_doc", "arguments": {}}</tool_call>', {}).tool_calls[0].name, "get_doc")

	def test_strict_history_uses_the_json_format(self):
		out = to_plain_messages(
			[
				{"role": "user", "content": "open 1"},
				{"role": "assistant", "content": None, "tool_calls": [
					{"id": "c1", "type": "function", "function": {"name": "get_doc", "arguments": '{"name": "1"}'}}
				]},
				{"role": "tool", "tool_call_id": "c1", "content": '{"status": "Open"}'},
				{"role": "assistant", "content": "It is open."},
			],
			[TOOL],
			strict=True,
		)
		self.assertIn('{"reply": ...}', out[0]["content"].replace('"<your answer>"', "..."))
		self.assertEqual(out[2]["content"], '{"tool": "get_doc", "arguments": {"name": "1"}}')
		self.assertTrue(out[3]["content"].startswith("Tool result:"))
		self.assertEqual(out[4]["content"], '{"reply": "It is open."}')


class TestPreloadHint(IntegrationTestCase):
	"""A greeting hands the panel what it needs to start the model download."""

	def test_hint_names_the_model_and_context_window(self):
		from flow.assistant.default import DEFAULT_MODEL_ID

		hint = webllm.preload_hint(DEFAULT_MODEL_ID)
		self.assertEqual(hint["model"], DEFAULT_MODEL_ID.split("/", 1)[1])
		self.assertGreater(hint["context_window_size"], 0)

	def test_no_hint_for_a_hosted_or_unknown_model(self):
		self.assertIsNone(webllm.preload_hint("openai/gpt-4o"))
		self.assertIsNone(webllm.preload_hint("webllm/not-a-flow-model"))
		self.assertIsNone(webllm.preload_hint(None))

	def test_greeting_carries_the_hint_and_thanks_does_not(self):
		from flow.tools.builtins import small_talk

		self.assertIn("preload", small_talk.func(kind="greeting"))
		self.assertNotIn("preload", small_talk.func(kind="thanks"))


class TestHiddenChoice(IntegrationTestCase):
	def test_the_chosen_word_comes_from_the_json_verdict_or_the_bare_word(self):
		from flow.lib.webllm import _chosen, _choice_schema

		self.assertEqual(_chosen('{"verdict": "OUT"}', ["IN", "OUT"]), "OUT")
		self.assertEqual(_chosen("IN", ["IN", "OUT"]), "IN")
		self.assertEqual(_chosen("Amsterdam", ["IN", "OUT"]), "")
		self.assertEqual(_choice_schema(["IN", "OUT"])["properties"]["verdict"]["enum"], ["IN", "OUT"])

	def test_a_choice_call_is_a_silent_request_limited_to_the_choices(self):
		from flow.lib.webllm import chat_stream

		request = next(chat_stream("webllm/some-model", [{"role": "user", "content": "hi"}], None, {}, ["IN", "OUT"]))
		self.assertTrue(request.silent)
		self.assertEqual(json.loads(request.response_schema)["properties"]["verdict"]["enum"], ["IN", "OUT"])
		plain = next(chat_stream("webllm/some-model", [{"role": "user", "content": "hi"}], None, {}))
		self.assertFalse(plain.silent)
