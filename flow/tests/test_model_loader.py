# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import threading
import unittest

from flow.lib.model_loader import LoadState, ModelFailed, ModelLoader, ModelLoading, Plan


class _Clock:
	def __init__(self):
		self.now = 100.0

	def __call__(self):
		return self.now


class _Thread:
	def __init__(self):
		self.alive = True

	def is_alive(self):
		return self.alive


class TestModelLoader(unittest.TestCase):
	"""The loader with a fake clock and a fake thread starter: no model, thread timing or sleep."""

	def setUp(self):
		self.clock = _Clock()
		self.jobs: list = []
		self.threads: list[_Thread] = []

		def spawn(run, name):
			self.jobs.append(run)
			thread = _Thread()
			self.threads.append(thread)
			return thread

		self.loader = ModelLoader(name="Test model", retry_seconds=300, clock=self.clock, spawn=spawn)

	def _plan(self, model="model", background=True, backend="fake"):
		return lambda: Plan(lambda: model, background=background, backend=backend)

	def test_unknown_key_is_not_loaded(self):
		status = self.loader.status("nothing")
		self.assertEqual(status.state, LoadState.NOT_LOADED)
		self.assertIsNone(status.error)

	def test_background_load_goes_not_loaded_loading_ready(self):
		with self.assertRaises(ModelLoading):
			self.loader.get("m", self._plan("the model"))
		self.clock.now += 5
		status = self.loader.status("m")
		self.assertEqual((status.state, status.seconds, status.backend), (LoadState.LOADING, 5.0, "fake"))

		self.jobs.pop()()
		status = self.loader.status("m")
		self.assertEqual(status.state, LoadState.READY)
		self.assertEqual(self.loader.get("m", self._plan("never used")), "the model")

	def test_acquire_returns_the_explanation_instead_of_raising(self):
		first = self.loader.acquire("m", self._plan())
		self.assertFalse(first.ready)
		self.assertIsNone(first.model)
		self.assertEqual(first.status.state, LoadState.LOADING)
		self.jobs.pop()()
		second = self.loader.acquire("m", self._plan())
		self.assertTrue(second.ready)
		self.assertEqual(second.model, "model")

	def test_requests_while_loading_start_one_load_and_do_not_replan(self):
		plans = []

		def plan():
			plans.append(1)
			return Plan(lambda: "model")

		for _ in range(5):
			with self.assertRaises(ModelLoading):
				self.loader.get("m", plan)
		self.assertEqual((len(plans), len(self.jobs)), (1, 1))

	def test_concurrent_requests_load_the_model_once(self):
		real = ModelLoader(retry_seconds=300)
		release, started = threading.Event(), threading.Event()
		loads = []

		def load():
			loads.append(1)
			started.set()
			release.wait(5)
			return "model"

		outcomes = []

		def request():
			try:
				outcomes.append(real.get("m", lambda: Plan(load)))
			except ModelLoading:
				outcomes.append("fallback")

		threads = [threading.Thread(target=request) for _ in range(12)]
		for t in threads:
			t.start()
		started.wait(5)
		for t in threads:
			t.join(5)
		self.assertEqual(outcomes.count("fallback"), 12)
		release.set()
		for _ in range(100):
			if real.status("m").state is LoadState.READY:
				break
			threading.Event().wait(0.05)
		self.assertEqual(len(loads), 1)
		self.assertEqual(real.get("m", lambda: Plan(load)), "model")
		self.assertEqual(len(loads), 1)

	def test_failed_load_waits_then_retries_and_recovers(self):
		def broken():
			raise MemoryError("no room")

		with self.assertRaises(ModelLoading):
			self.loader.get("m", lambda: Plan(broken))
		self.jobs.pop()()
		status = self.loader.status("m")
		self.assertEqual(status.state, LoadState.FAILED)
		self.assertEqual(status.error, "MemoryError: no room")
		self.assertEqual(status.retry_in, 300.0)

		self.clock.now += 100
		with self.assertRaises(ModelFailed) as caught:
			self.loader.get("m", self._plan())
		self.assertEqual(caught.exception.status.retry_in, 200.0)
		self.assertIn("MemoryError: no room", str(caught.exception))
		self.assertEqual(len(self.jobs), 0)

		self.clock.now += 201
		with self.assertRaises(ModelLoading):
			self.loader.get("m", self._plan("recovered"))
		self.jobs.pop()()
		self.assertEqual(self.loader.get("m", self._plan()), "recovered")
		self.assertIsNone(self.loader.status("m").error)

	def test_the_plan_is_made_again_on_each_retry(self):
		backends = iter(["laya", "onnx"])

		def plan():
			return Plan(lambda: "model", backend=next(backends))

		with self.assertRaises(ModelLoading):
			self.loader.get("m", plan)
		self.assertEqual(self.loader.status("m").backend, "laya")
		self.threads[-1].alive = False  # the thread was killed without a result
		self.assertEqual(self.loader.status("m").state, LoadState.FAILED)

		self.clock.now += 301
		with self.assertRaises(ModelLoading):
			self.loader.get("m", plan)
		self.assertEqual(self.loader.status("m").backend, "onnx")

	def test_a_thread_that_dies_without_a_result_is_a_failure_not_loading_forever(self):
		with self.assertRaises(ModelLoading):
			self.loader.get("m", self._plan())
		self.threads[-1].alive = False
		status = self.loader.status("m")
		self.assertEqual(status.state, LoadState.FAILED)
		self.assertEqual(status.error, "load thread exited without a result")
		with self.assertRaises(ModelFailed):
			self.loader.get("m", self._plan())

	def test_a_load_that_ends_without_a_result_is_a_failure(self):
		def quits():
			raise SystemExit

		with self.assertRaises(ModelLoading):
			self.loader.get("m", lambda: Plan(quits))
		with self.assertRaises(SystemExit):
			self.jobs.pop()()
		self.assertEqual(self.loader.status("m").state, LoadState.FAILED)

	def test_a_load_that_is_not_in_the_background_runs_in_the_request(self):
		self.assertEqual(self.loader.get("m", self._plan("torch model", background=False)), "torch model")
		self.assertEqual(self.jobs, [])
		self.assertEqual(self.loader.status("m").state, LoadState.READY)

	def test_a_failed_load_in_the_request_raises_its_own_error_then_waits(self):
		def broken():
			raise ImportError("no laya")

		with self.assertRaises(ImportError):
			self.loader.get("m", lambda: Plan(broken, background=False))
		self.assertEqual(self.loader.status("m").state, LoadState.FAILED)
		with self.assertRaises(ModelFailed):
			self.loader.get("m", self._plan(background=False))
		self.clock.now += 301
		self.assertEqual(self.loader.get("m", self._plan("ok", background=False)), "ok")

	def test_a_plan_that_raises_is_recorded_and_raised(self):
		def plan():
			raise OSError("offline")

		with self.assertRaises(OSError):
			self.loader.get("m", plan)
		status = self.loader.status("m")
		self.assertEqual((status.state, status.error), (LoadState.FAILED, "OSError: offline"))
		self.assertEqual(self.jobs, [])

	def test_models_do_not_share_state(self):
		with self.assertRaises(ModelLoading):
			self.loader.get("a", self._plan("A"))
		self.assertEqual(self.loader.get("b", self._plan("B", background=False)), "B")
		self.assertEqual(self.loader.status("a").state, LoadState.LOADING)
		self.assertEqual(self.loader.status("b").state, LoadState.READY)
