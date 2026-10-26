# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Loads each model once per worker and says plainly what state it is in.

A model is `not_loaded`, `loading`, `ready` or `failed`. `ModelLoader.get` returns a ready model or
raises `ModelLoading` (a load is under way) or `ModelFailed` (the last attempt failed and the retry
wait hasn't passed); `status` reports the same thing without starting anything. `acquire` is
`get` that returns the explanation instead of raising.

State lives in a `ModelLoader` instance, so it is per worker process, never shared between
workers, and a test makes its own loader instead of resetting globals. Nothing here knows about
Laya or Frappe.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

RETRY_SECONDS = 300


class LoadState(str, enum.Enum):
	NOT_LOADED = "not_loaded"
	LOADING = "loading"
	READY = "ready"
	FAILED = "failed"


class ModelLoading(Exception):
	"""The model isn't loaded yet; the caller uses its fallback."""


class ModelFailed(RuntimeError):
	"""The last load failed and the retry wait hasn't passed. `status` says why."""

	def __init__(self, message: str, status: LoadStatus):
		super().__init__(message)
		self.status = status


@dataclass(frozen=True)
class Plan:
	"""How to load a model, decided only when a load starts (so it is decided again on each retry):
	`load` builds it, `background` runs that in a thread instead of the request, and `backend` is a
	label for status."""

	load: Callable[[], Any]
	background: bool = True
	backend: str | None = None


@dataclass(frozen=True)
class LoadStatus:
	key: str
	state: LoadState = LoadState.NOT_LOADED
	error: str | None = None
	backend: str | None = None
	seconds: float | None = None  # elapsed while loading, the load's duration once it ended
	retry_in: float | None = None  # until a failed load is tried again


@dataclass(frozen=True)
class Acquired:
	"""A ready model, or why there isn't one."""

	model: Any
	status: LoadStatus

	@property
	def ready(self) -> bool:
		return self.status.state is LoadState.READY


class _Entry:
	def __init__(self, key: str, now: float):
		self.key = key
		self.state = LoadState.LOADING
		self.model: Any = None
		self.error: str | None = None
		self.backend: str | None = None
		self.started = now
		self.finished: float | None = None
		self.retry_at: float | None = None
		self.thread: Any = None


def _start_thread(run: Callable[[], None], name: str) -> threading.Thread:
	thread = threading.Thread(target=run, name=name, daemon=True)
	thread.start()
	return thread


class ModelLoader:
	def __init__(
		self,
		*,
		name: str = "Model",
		retry_seconds: float = RETRY_SECONDS,
		clock: Callable[[], float] = time.monotonic,
		spawn: Callable[[Callable[[], None], str], Any] = _start_thread,
		logger: logging.Logger | None = None,
	):
		self.name = name
		self.retry_seconds = retry_seconds
		self._clock = clock
		self._spawn = spawn
		self._logger = logger or logging.getLogger("flow.laya")
		self._lock = threading.Lock()  # guards _entries; never held while a model loads
		self._entries: dict[str, _Entry] = {}

	def get(self, key: str, plan: Callable[[], Plan]) -> Any:
		"""The ready model for `key`. `plan` runs only when a load has to start. Raises ModelLoading
		while one is under way, ModelFailed during the retry wait after a failure, and the loader's
		own exception when a load that ran in this request fails."""
		result = self.acquire(key, plan)
		if result.ready:
			return result.model
		if result.status.state is LoadState.FAILED:
			raise ModelFailed(f"{self.name} failed to load: {result.status.error}", result.status)
		raise ModelLoading(f"{self.name} is still loading in the background; the chat model answers meanwhile")

	def acquire(self, key: str, plan: Callable[[], Plan]) -> Acquired:
		"""Like `get`, but returns the model or its status. A load that runs in this request and
		fails still raises its own exception."""
		with self._lock:
			entry = self._entries.get(key)
			if entry is not None:
				self._reap(entry)
				if entry.state is LoadState.READY:
					return Acquired(entry.model, self._status(entry))
				if entry.state is LoadState.LOADING or not self._retry_due(entry):
					return Acquired(None, self._status(entry))
			# Claimed here, so concurrent requests see LOADING and fall back while this one plans.
			entry = self._entries[key] = _Entry(key, self._clock())

		try:
			chosen = plan()
		except Exception as exc:
			self._fail(entry, exc)
			raise
		entry.backend = chosen.backend

		if chosen.background:
			try:
				entry.thread = self._spawn(lambda: self._run(entry, chosen.load), f"flow-load-{key}")
			except Exception as exc:
				self._fail(entry, exc)
				raise
			return Acquired(None, self.status(key))

		try:
			model = chosen.load()
		except Exception as exc:
			self._fail(entry, exc)
			raise
		self._ready(entry, model)
		return Acquired(model, self.status(key))

	def status(self, key: str) -> LoadStatus:
		"""What this worker knows about `key`, without starting a load."""
		with self._lock:
			entry = self._entries.get(key)
			if entry is None:
				return LoadStatus(key)
			self._reap(entry)
			return self._status(entry)

	def _run(self, entry: _Entry, load: Callable[[], Any]) -> None:
		try:
			model = load()
		except Exception as exc:
			self._fail(entry, exc)
		else:
			self._ready(entry, model)
		finally:
			with self._lock:
				if entry.state is LoadState.LOADING:
					self._mark_failed(entry, "load ended without a result")

	def _ready(self, entry: _Entry, model: Any) -> None:
		with self._lock:
			entry.model, entry.state, entry.error = model, LoadState.READY, None
			entry.finished = self._clock()
			seconds = entry.finished - entry.started
		self._logger.info("%s %s ready in %.1fs", self.name, entry.key, seconds)

	def _fail(self, entry: _Entry, exc: BaseException) -> None:
		message = f"{type(exc).__name__}: {exc}"[:900]
		with self._lock:
			self._mark_failed(entry, message)
		self._logger.warning("%s %s failed to load: %s", self.name, entry.key, message)

	def _mark_failed(self, entry: _Entry, message: str) -> None:
		entry.state, entry.error = LoadState.FAILED, message
		entry.finished = self._clock()
		entry.retry_at = entry.finished + self.retry_seconds

	def _reap(self, entry: _Entry) -> None:
		"""A background load whose thread is gone without a result (killed, say) has failed."""
		thread = entry.thread
		if entry.state is LoadState.LOADING and thread is not None and not thread.is_alive():
			self._mark_failed(entry, "load thread exited without a result")

	def _retry_due(self, entry: _Entry) -> bool:
		return entry.retry_at is not None and self._clock() >= entry.retry_at

	def _status(self, entry: _Entry) -> LoadStatus:
		now = self._clock()
		retry_in = None
		if entry.state is LoadState.FAILED and entry.retry_at is not None:
			retry_in = max(0.0, entry.retry_at - now)
		seconds = (entry.finished if entry.finished is not None else now) - entry.started
		return LoadStatus(
			entry.key, entry.state, entry.error, entry.backend, round(seconds, 1), retry_in and round(retry_in, 1)
		)
