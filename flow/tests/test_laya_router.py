# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import json
from unittest.mock import MagicMock, patch

import frappe
from frappe.tests import IntegrationTestCase

from flow.lib import laya_router

TOOLS = {"count", "creation_steps", "assignment_steps", "status_steps", "error_diagnosis", "small_talk", "required_values", "create", "update"}


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

	def test_unreachable_hub_is_an_error_not_a_cached_answer(self):
		laya_router._backends.pop("finetuned:owner/repo", None)
		with patch("huggingface_hub.HfApi.list_repo_files", side_effect=OSError("offline")):
			with self.assertRaises(OSError):
				laya_router._finetuned_agent("owner/repo")
		self.assertNotIn("finetuned:owner/repo", laya_router._backends)

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

	def test_model_loads_in_background_without_blocking(self):
		import threading

		release = threading.Event()

		def slow_load():
			release.wait(5)
			return "model"

		key = "test:background"
		laya_router._models.pop(key, None)
		laya_router._loading.pop(key, None)
		try:
			with self.assertRaises(laya_router.ModelLoading):
				laya_router._model(key, slow_load)
			release.set()
			for _ in range(50):
				if key in laya_router._models:
					break
				threading.Event().wait(0.05)
			self.assertEqual(laya_router._model(key, slow_load), "model")
		finally:
			laya_router._models.pop(key, None)
			laya_router._loading.pop(key, None)

	def test_loading_model_falls_back_quietly(self):
		with patch.object(laya_router, "_router", side_effect=laya_router.ModelLoading("still loading")):
			self.assertIsNone(laya_router.route("my invoice won't submit", TOOLS))
		log = self._last_log()
		self.assertEqual((log.outcome, log.reason), ("Fallback", "still loading"))

	def test_pytorch_models_load_in_the_request(self):
		key = "test:foreground"
		laya_router._models.pop(key, None)
		try:
			self.assertEqual(laya_router._model(key, lambda: "model", background=False), "model")
			self.assertNotIn(key, laya_router._loading)
		finally:
			laya_router._models.pop(key, None)

