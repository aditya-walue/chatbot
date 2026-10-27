# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import json
from unittest.mock import MagicMock, patch

import frappe
from frappe.tests import IntegrationTestCase

from flow.lib import laya_router
from flow.lib.model_loader import LoadState, ModelFailed, ModelLoader
from flow.lib.tool import LayaRoute, tool

TOOLS = {"count", "creation_steps", "assignment_steps", "status_steps", "docs_howto", "error_diagnosis", "small_talk", "required_values", "create", "update"}


def _laya(intent, confidence, asks_count=0.0):
	"""A fake Laya Router returning one fixed prediction."""
	router = MagicMock()
	router.predict.return_value = {
		"answers": {
			"intent": {
				"type": "choice",
				"choice": intent,
				"probabilities": {intent: confidence, "OTHER": round(1 - confidence, 4)},
				"confidence": confidence,
				"answer_confidence": confidence,
			},
			"asks_count": {"type": "noul", "noul": asks_count, "answer_confidence": max(asks_count, 1 - asks_count)},
			"complexity": {"type": "score", "score": 1.0, "answer_confidence": 0.5},
		},
		"routing": {"model": "english"},
	}
	return router


class _Clock:
	def __init__(self):
		self.now = 1000.0

	def __call__(self):
		return self.now


class _Thread:
	def __init__(self, alive=True):
		self.alive = alive

	def is_alive(self):
		return self.alive


class TestLayaRouter(IntegrationTestCase):
	def setUp(self):
		self._conf = {
			k: frappe.conf.get(k)
			for k in (
				"flow_laya_enabled",
				"flow_laya_threshold",
				"flow_laya_model",
				"flow_laya_model_path",
				"flow_laya_backend",
			)
		}
		# These tests stub Laya; a site pointed at a real checkpoint must not load it here.
		frappe.conf.flow_laya_model_path = None
		frappe.conf.flow_laya_model = "english"
		frappe.conf.flow_laya_backend = None
		frappe.conf.flow_laya_enabled = True
		frappe.conf.flow_laya_threshold = 0.8
		# Each test gets its own loader, so none depends on what another loaded in this process.
		self.clock, self.jobs = _Clock(), []
		self.loader = ModelLoader(
			name="Laya model",
			retry_seconds=300,
			clock=self.clock,
			spawn=lambda run, name: self.jobs.append(run) or _Thread(),
		)
		patcher = patch.object(laya_router, "_loader", self.loader)
		patcher.start()
		self.addCleanup(patcher.stop)

	def tearDown(self):
		for k, v in self._conf.items():
			frappe.conf[k] = v
		frappe.db.rollback()

	def _route(self, router, text, tools=TOOLS):
		with patch.object(laya_router, "_router", return_value=router):
			return laya_router.route(text, tools)

	def _last_log(self):
		return frappe.get_last_doc("Flow Route Log")

	def test_status_reports_settings_without_the_token(self):
		frappe.conf.flow_laya_enabled = True
		frappe.conf.flow_laya_model_path = "owner/some-checkpoint"
		frappe.conf.flow_hf_token = "hf_secret_value"
		status = laya_router.status()
		self.assertTrue(status["enabled"])
		self.assertEqual(status["model_path"], "owner/some-checkpoint")
		self.assertTrue(status["token_set"])
		self.assertNotIn("hf_secret_value", json.dumps(status, default=str))
		self.assertFalse(status["loaded_in_this_worker"])

	def test_on_with_flows_checkpoint_when_site_config_is_empty(self):
		frappe.conf.flow_laya_enabled = None
		frappe.conf.flow_laya_threshold = None
		frappe.conf.flow_laya_model = None
		frappe.conf.flow_laya_model_path = None
		status = laya_router.status()
		self.assertTrue(status["enabled"])
		self.assertEqual(status["model_path"], laya_router.DEFAULT_MODEL_PATH)
		self.assertEqual(status["threshold"], laya_router.DEFAULT_THRESHOLD)

	def test_site_can_still_turn_it_off(self):
		frappe.conf.flow_laya_enabled = 0
		self.assertFalse(laya_router.enabled())

	def test_status_endpoint_is_for_system_managers(self):
		from flow.api import laya_status

		self.assertIn("enabled", laya_status())
		with self.set_user("Guest"), self.assertRaises(frappe.PermissionError):
			laya_status()

	def test_disabled_does_nothing(self):
		frappe.conf.flow_laya_enabled = False
		router = _laya("TROUBLESHOOTING", 0.99)
		self.assertIsNone(self._route(router, "Something exploded"))
		router.predict.assert_not_called()

	def test_confident_troubleshooting_routes_to_diagnosis_and_logs(self):
		result = self._route(_laya("TROUBLESHOOTING", 0.93), "Something exploded on save")
		self.assertEqual(result, ("error_diagnosis", {"error": "Something exploded on save"}))
		log = self._last_log()
		self.assertEqual((log.outcome, log.intent, log.tool), ("Routed", "TROUBLESHOOTING", "error_diagnosis"))
		self.assertAlmostEqual(log.confidence, 0.93)
		self.assertEqual(json.loads(log.distribution)["TROUBLESHOOTING"], 0.93)

	def test_low_confidence_falls_back_to_the_chat_model(self):
		self.assertIsNone(self._route(_laya("HOWTO", 0.4), "how do I add a todo"))
		log = self._last_log()
		self.assertEqual(log.outcome, "Fallback")
		self.assertIn("below", log.reason)

	def test_howto_and_count_need_a_doctype(self):
		self.assertEqual(self._route(_laya("HOWTO", 0.9), "guide me through adding a todo"), ("creation_steps", {"doctype": "ToDo"}))
		self.assertIsNone(self._route(_laya("HOWTO", 0.9), "guide me through life"))

	def test_howto_add_to_goes_to_the_docs_when_no_site_tool_fits(self):
		result = self._route(_laya("HOWTO", 0.91), "how to add new tax component to the sales invoice?")
		self.assertEqual(result, ("docs_howto", {"topic": "add new tax component sales invoice"}))
		log = self._last_log()
		self.assertEqual((log.outcome, log.tool), ("Routed", "docs_howto"))

	def test_howto_add_to_keeps_the_site_tools_first(self):
		# Creating a record and linking one record to another still use their own tools.
		self.assertEqual(self._route(_laya("HOWTO", 0.92), "how do I add a new todo")[0], "creation_steps")
		self.assertEqual(self._route(_laya("HOWTO", 0.92), "how do I add a role to a user")[0], "assignment_steps")

	def test_howto_add_to_needs_confidence_and_the_tool(self):
		text = "how to add new tax component to the sales invoice?"
		self.assertIsNone(self._route(_laya("HOWTO", 0.5), text))
		self.assertIn("below", self._last_log().reason)
		# An agent without the tool falls back to the chat model, and the log says why.
		self.assertIsNone(self._route(_laya("HOWTO", 0.95), text, tools=TOOLS - {"docs_howto"}))
		self.assertIn("docs_howto isn't enabled", self._last_log().reason)

	def test_only_howto_intent_reaches_the_docs_route(self):
		text = "how to add new tax component to the sales invoice?"
		self.assertIsNone(self._route(_laya("DEFINITION", 0.95), text))
		self.assertIsNone(self._route(_laya("WORKFLOW", 0.95), text))

	def test_howto_that_is_not_add_to_still_falls_back(self):
		self.assertIsNone(self._route(_laya("HOWTO", 0.95), "how does a purchase order get cancelled"))
		self.assertIn("no direct Flow path", self._last_log().reason)

	def test_the_tool_map_is_declared_on_the_tools(self):
		from flow.tools import builtins

		known = set(laya_router.INTENTS) | set(laya_router.FLOW_INTENTS)
		declared = {t.name: t.laya.intent for t in builtins.BUILTIN_TOOLS if t.laya}
		self.assertTrue(declared)
		self.assertLessEqual(set(declared.values()), known)

	def test_a_tool_that_declares_an_intent_is_routed_to_without_touching_the_router(self):
		from flow.tools import builtins

		@tool(laya=LayaRoute("DEFINITION", lambda text, signals: {"word": text}))
		def define(word: str) -> str:
			"""Define a word."""
			return word

		with patch.object(builtins, "BUILTIN_TOOLS", [*builtins.BUILTIN_TOOLS, define]):
			self.assertEqual(self._route(_laya("DEFINITION", 0.9), "what is a widget", TOOLS | {"define"}), ("define", {"word": "what is a widget"}))
			self.assertIsNone(self._route(_laya("DEFINITION", 0.9), "what is a widget", TOOLS))
			self.assertIn("define isn't enabled", self._last_log().reason)

	def test_the_first_tool_that_reads_the_message_takes_it(self):
		from flow.tools import builtins

		@tool(laya=LayaRoute("DEFINITION", lambda text, signals: None))
		def passes(word: str) -> str:
			"""Passes."""
			return word

		@tool(laya=LayaRoute("DEFINITION", lambda text, signals: {"word": "second"}))
		def takes(word: str) -> str:
			"""Takes."""
			return word

		with patch.object(builtins, "BUILTIN_TOOLS", [passes, takes]):
			self.assertEqual(self._route(_laya("DEFINITION", 0.9), "anything", {"passes", "takes"}), ("takes", {"word": "second"}))

	def test_a_tool_sees_how_sure_laya_was_that_the_message_asks_for_a_count(self):
		seen = []

		def read_signals(text, signals):
			seen.append((signals.asks_count, signals.threshold))

		from flow.tools import builtins

		@tool(laya=LayaRoute("RECORD_LOOKUP", read_signals))
		def peek(word: str) -> str:
			"""Peek."""
			return word

		with patch.object(builtins, "BUILTIN_TOOLS", [peek]):
			self._route(_laya("RECORD_LOOKUP", 0.9, asks_count=0.7), "how many things", {"peek"})
		self.assertEqual(seen, [(0.7, 0.8)])

	def test_howto_routes_only_requests_to_create(self):
		self.assertEqual(self._route(_laya("HOWTO", 0.9), "walk me through setting up a new todo"), ("creation_steps", {"doctype": "ToDo"}))
		self.assertEqual(self._route(_laya("HOWTO", 0.9), "what are the steps for making a note?"), ("creation_steps", {"doctype": "Note"}))
		# How-to questions about doing something else with a DocType go to the chat model.
		self.assertEqual(
			self._route(_laya("HOWTO", 0.92), "could you tell me how to get this todo marked as closed"),
			("status_steps", {"doctype": "ToDo", "status": "Closed"}),
		)
		self.assertEqual(
			self._route(_laya("HOWTO", 0.92), "can you tell me the way to give a role to a user"),
			("assignment_steps", {"item": "Role", "target": "User"}),
		)
		for text in (
			"how do I link my car to the moon",
			"how to approve a leave application",
		):
			self.assertIsNone(self._route(_laya("HOWTO", 0.92), text), text)
			self.assertIn("no direct Flow path", self._last_log().reason)
		self.assertEqual(
			self._route(_laya("RECORD_LOOKUP", 0.9, asks_count=0.95), "number of todos we have"),
			("count", {"doctype": "ToDo"}),
		)
		# A lookup that isn't a count goes to the chat model, which picks read's fields.
		self.assertIsNone(self._route(_laya("RECORD_LOOKUP", 0.9, asks_count=0.1), "show open todos"))

	def test_never_routes_to_a_write(self):
		# A confident CREATE with values must reach the model, prechecks and approval card.
		self.assertIsNone(self._route(_laya("CREATE", 0.99), "create a todo with description call vendor"))
		# Without values it only asks for them (read-only).
		self.assertEqual(self._route(_laya("CREATE", 0.99), "create a todo"), ("required_values", {"doctype": "ToDo"}))

	def test_unavailable_tool_falls_back(self):
		self.assertIsNone(self._route(_laya("TROUBLESHOOTING", 0.95), "it broke", tools={"count"}))
		self.assertIn("isn't enabled", self._last_log().reason)

	def test_laya_failure_is_logged_and_ignored(self):
		router = MagicMock()
		router.predict.side_effect = ImportError("No module named 'laya'")
		self.assertIsNone(self._route(router, "how many todos"))
		log = self._last_log()
		self.assertEqual(log.outcome, "Error")
		self.assertIn("ImportError", log.error)

	def test_metrics_summarise_outcomes(self):
		frappe.db.delete("Flow Route Log")
		self._route(_laya("TROUBLESHOOTING", 0.95), "it broke")
		self._route(_laya("HOWTO", 0.3), "how do I add a todo")
		self._route(_laya("HOWTO", 0.2), "how do I add a note")
		m = laya_router.metrics(days=1)
		self.assertEqual((m["total"], m["routed"], m["fallback"], m["errors"]), (3, 1, 2, 0))
		self.assertAlmostEqual(m["fallback_rate"], 0.6667, places=3)
		self.assertEqual(m["routed_by_intent"], {"TROUBLESHOOTING": 1})

	def test_agent_uses_laya_when_no_exact_route_matches(self):
		from flow.lib.agent import Agent
		from flow.tools.builtins import error_diagnosis

		model = MagicMock()
		with patch.object(laya_router, "_router", return_value=_laya("TROUBLESHOOTING", 0.95)):
			result = Agent(model=model, tools=[error_diagnosis]).run("my sales order keeps exploding when I save it")
		model.chat.assert_not_called()
		self.assertIn("**Error diagnosis**", result.output)

	def test_finetuned_checkpoint_gets_the_questions_it_was_trained_on(self):
		agent = MagicMock()
		agent.system_one.return_value = _laya("TROUBLESHOOTING", 0.95).predict.return_value
		frappe.conf.flow_laya_model_path = "/tmp/flow-laya-model"
		try:
			with patch.object(laya_router, "_finetuned_agent", return_value=agent):
				result = laya_router.route("something exploded", TOOLS)
		finally:
			frappe.conf.flow_laya_model_path = None
		self.assertEqual(result[0], "error_diagnosis")
		self.assertEqual(agent.system_one.call_args[0][1], laya_router.FINETUNED_QUESTIONS)

	def test_onnx_checkpoint_uses_flows_runner(self):
		import tempfile

		from flow.lib.laya_onnx import ONNX_FILE

		with tempfile.TemporaryDirectory() as folder:
			self.assertFalse(laya_router._use_onnx(folder))
			open(f"{folder}/{ONNX_FILE}", "w").close()
			self.assertTrue(laya_router._use_onnx(folder))
			self.assertFalse(laya_router._use_onnx(folder, backend="laya"))

	def test_unreachable_hub_is_an_error_then_a_failed_state_not_a_cached_answer(self):
		with patch("huggingface_hub.HfApi.list_repo_files", side_effect=OSError("offline")):
			with self.assertRaises(OSError):
				laya_router._finetuned_agent("owner/repo")
			with self.assertRaises(ModelFailed):
				laya_router._finetuned_agent("owner/repo")
		self.assertEqual(self.loader.status("finetuned:owner/repo").state, LoadState.FAILED)

	def test_backend_is_decided_again_when_a_failed_load_is_retried(self):
		model = MagicMock()
		with patch.object(laya_router, "_use_onnx", side_effect=[OSError("offline"), True]) as probe:
			with self.assertRaises(OSError):
				laya_router._finetuned_agent("owner/repo")
			self.clock.now += 301
			with self.assertRaises(laya_router.ModelLoading):
				laya_router._finetuned_agent("owner/repo")
			self.assertEqual(probe.call_count, 2)
		with patch("flow.lib.laya_onnx.OnnxLaya", return_value=model):
			self.jobs.pop()()
		self.assertIs(laya_router._finetuned_agent("owner/repo"), model)

	def test_status_before_during_and_after_loading(self):
		frappe.conf.flow_laya_model_path = "owner/repo"
		model = MagicMock()
		status = laya_router.status()
		self.assertEqual((status["load_state"], status["loaded_in_this_worker"]), ("not_loaded", False))
		self.assertEqual(status["scope"], "this worker only")
		self.assertIsInstance(status["worker_pid"], int)

		with patch.object(laya_router, "_use_onnx", return_value=True):
			with self.assertRaises(laya_router.ModelLoading):
				laya_router._finetuned_agent("owner/repo")
		self.clock.now += 12
		status = laya_router.status()
		self.assertEqual((status["load_state"], status["loaded_in_this_worker"]), ("loading", False))
		self.assertEqual((status["load_seconds"], status["onnx_checkpoint"]), (12.0, True))

		with patch("flow.lib.laya_onnx.OnnxLaya", return_value=model):
			self.jobs.pop()()
		status = laya_router.status()
		self.assertEqual((status["load_state"], status["loaded_in_this_worker"]), ("ready", True))
		self.assertIsNone(status["loading_error"])

	def test_status_reports_a_failed_load_and_recovers_after_the_retry_wait(self):
		frappe.conf.flow_laya_model_path = "owner/repo"
		with patch.object(laya_router, "_use_onnx", return_value=True):
			with self.assertRaises(laya_router.ModelLoading):
				laya_router._finetuned_agent("owner/repo")
			with patch("flow.lib.laya_onnx.OnnxLaya", side_effect=MemoryError("no room")):
				self.jobs.pop()()
			status = laya_router.status()
			self.assertEqual(status["load_state"], "failed")
			self.assertIn("MemoryError", status["loading_error"])
			self.assertEqual(status["retry_in_seconds"], 300.0)
			self.assertFalse(status["loaded_in_this_worker"])
			with self.assertRaises(ModelFailed):
				laya_router._finetuned_agent("owner/repo")

			self.clock.now += 301
			with self.assertRaises(laya_router.ModelLoading):
				laya_router._finetuned_agent("owner/repo")
			with patch("flow.lib.laya_onnx.OnnxLaya", return_value="model"):
				self.jobs.pop()()
		self.assertEqual(laya_router.status()["load_state"], "ready")
		self.assertEqual(laya_router._finetuned_agent("owner/repo"), "model")

	def test_a_failed_model_is_logged_as_an_error_and_falls_back(self):
		frappe.conf.flow_laya_model_path = "owner/repo"
		frappe.conf.flow_laya_model = None
		with patch.object(laya_router, "_use_onnx", side_effect=OSError("offline")):
			self.assertIsNone(laya_router.route("my invoice won't submit", TOOLS))
		log = self._last_log()
		self.assertEqual(log.outcome, "Error")
		self.assertIn("offline", log.error)

	def test_hub_probe_that_hangs_gives_up(self):
		import time

		def hang(*args, **kwargs):
			time.sleep(2)

		with patch.object(laya_router, "HF_PROBE_SECONDS", 0.2), patch(
			"huggingface_hub.HfApi.list_repo_files", side_effect=hang
		):
			with self.assertRaises(TimeoutError):
				laya_router._use_onnx("owner/repo")

	def test_prefetch_downloads_a_hub_checkpoint_only_when_enabled(self):
		frappe.conf.flow_laya_model = None
		with patch("flow.lib.laya_onnx._resolve") as resolve:
			frappe.conf.flow_laya_enabled = False
			laya_router.prefetch()
			resolve.assert_not_called()

			frappe.conf.flow_laya_enabled = True
			frappe.conf.flow_laya_model_path = "/tmp/local-checkpoint"
			with patch("os.path.isdir", return_value=True):
				laya_router.prefetch()
			resolve.assert_not_called()

			frappe.conf.flow_laya_model_path = "owner/repo"
			laya_router.prefetch()
			resolve.assert_called_once_with("owner/repo", laya_router._hf_token())

	def test_prefetch_never_raises(self):
		frappe.conf.flow_laya_model = None
		frappe.conf.flow_laya_model_path = "owner/repo"
		with patch("flow.lib.laya_onnx._resolve", side_effect=OSError("offline")):
			laya_router.prefetch()

	def test_hf_token_from_config_when_env_missing(self):
		with patch.dict("os.environ", {}, clear=False):
			import os

			os.environ.pop("HF_TOKEN", None)
			frappe.conf.flow_hf_token = "test-token"
			try:
				self.assertEqual(laya_router._hf_token(), "test-token")
				os.environ["HF_TOKEN"] = "env-token"
				self.assertEqual(laya_router._hf_token(), "env-token")
			finally:
				frappe.conf.flow_hf_token = None

	def test_loading_model_falls_back_quietly(self):
		with patch.object(laya_router, "_router", side_effect=laya_router.ModelLoading("still loading")):
			self.assertIsNone(laya_router.route("my invoice won't submit", TOOLS))
		log = self._last_log()
		self.assertEqual((log.outcome, log.reason), ("Fallback", "still loading"))
