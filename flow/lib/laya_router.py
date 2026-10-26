# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Optional Laya decision layer: classify a user message before the chat model sees it.

Laya (https://github.com/NandhaKishorM/laya) is a small non-autoregressive classifier that
answers typed questions (`choice`, `score`, `noul`) about a text in one forward pass, with a
calibrated confidence. Flow asks it which kind of request a message is. When it is confident and
the request maps to a read-only Flow tool, the agent runs that tool directly; otherwise the chat
model (e.g. Qwen in the browser) handles the message exactly as before.

Laya only chooses. It never calls create/update/delete or any business action: the tools it can
route to are read-only, and writes still go through the model, the tools' prechecks, the user's
permissions and the approval card.

On by default with Flow's own checkpoint, and optional: if the checkpoint or runtime isn't
available, fails to load, or errors, Flow behaves as if it were disabled. Every setting below is
an optional override in site config (site_config.json):

    flow_laya_enabled    set to 0/false to turn it off (default on)
    flow_laya_threshold  calibrated confidence needed to route directly (default 0.85)
    flow_laya_model      Laya checkpoint name, e.g. "english" (default: Laya's own routing)
    flow_laya_model_path checkpoint fine-tuned on Flow (scripts/laya/finetune.py): a local folder or
                         a Hugging Face repo id (default DEFAULT_MODEL_PATH); asked FINETUNED_QUESTIONS, the questions it was
                         trained on. A checkpoint containing laya.onnx runs on Flow's own ONNX
                         runner (flow/lib/laya_onnx.py) — no PyTorch or `laya` package needed
    flow_hf_token        read token for a private Hugging Face checkpoint when the HF_TOKEN
                         environment variable can't be set (e.g. Frappe Cloud bench config)
    flow_laya_backend    "onnx" or "laya" to force the runtime for flow_laya_model_path (default:
                         ONNX when the checkpoint has laya.onnx)

Every decision is logged as a Flow Route Log row (see `metrics` for the summary).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import frappe

DEFAULT_THRESHOLD = 0.85
HF_PROBE_SECONDS = 10
# Flow's fine-tuned checkpoint, int8-quantized: about 440 MB and 600 MB of RAM per worker, against
# 1.7 GB and 1.6 GB for the fp32 export in aditya-walue/flow-laya-intent. Laya then works with no
# site config.
DEFAULT_MODEL_PATH = "aditya-walue/flow-laya-intent-int8"
INPUT_LOG_LIMIT = 500

# The intents shared with the other assistants on this bench, plus the two Flow also handles.
INTENTS: dict[str, str] = {
	"HOWTO": "asks how to create, add or set up a record by hand, or for the steps to do it",
	"RECORD_LOOKUP": "asks to count, find, list or show existing records or their details",
	"TROUBLESHOOTING": "reports an error message, a failure, or something that is not working",
	"CREATE": "asks the assistant to create or add a new record now",
	"DEFINITION": "asks what a term, document type or field means or is used for",
	"WORKFLOW": "asks about a business process with several steps, approvals or documents",
	"SMALL_TALK": "a greeting, thanks or casual chat with no request",
}

QUESTIONS: dict[str, dict[str, Any]] = {
	"intent": {
		"type": "choice",
		"instructions": "What kind of request is this message to an ERP assistant?",
		"criteria": INTENTS,
	},
	"asks_count": {
		"type": "noul",
		"instructions": "Does the message ask how many records there are (a count or number)?",
	},
	"complexity": {
		"type": "score",
		"instructions": "How complex is the request?",
		"criteria": ["one simple request", "a request with a couple of parts", "a complex multi-step request"],
	},
}
# Questions for a Laya checkpoint fine-tuned on Flow (scripts/laya/finetune.py trains on exactly
# these, so they must not change without retraining). Five intents only: create requests and
# greetings are handled by exact routes before Laya is asked.
FLOW_INTENTS: dict[str, str] = {
	"HOWTO": "asks how to create, add or set up a record by hand, or for the steps to do it",
	"DEFINITION": "asks what a term, document type or field means or is used for",
	"WORKFLOW": "asks about a business process with several steps, approvals or documents",
	"RECORD_LOOKUP": "asks to count, find, list or show existing records or their details",
	"TROUBLESHOOTING": "reports an error message, a failure, or something that is not working",
}
FINETUNED_QUESTIONS: dict[str, dict[str, Any]] = {
	"intent": {
		"type": "choice",
		"instructions": "What kind of request is this message to an ERP assistant?",
		"criteria": FLOW_INTENTS,
	},
	"asks_count": {
		"type": "noul",
		"instructions": "Does the message ask how many records there are (a count or number)?",
	},
}

# `complexity` (0 = simple .. 2 = complex) is logged for analysis but not used to gate: on the
# shipped checkpoint it scores nearly every message about 1.0, so it carries no signal yet.


@dataclass
class RouteDecision:
	"""Laya's verdict on one message, and what Flow did with it."""

	intent: str | None = None
	confidence: float = 0.0
	distribution: dict[str, float] = field(default_factory=dict)
	asks_count: float | None = None
	complexity: float | None = None
	model: str | None = None
	outcome: str = "Fallback"  # Routed | Fallback | Error
	reason: str = ""
	tool: str | None = None
	arguments: dict[str, Any] | None = None
	latency_ms: int = 0
	error: str | None = None


def enabled() -> bool:
	"""On unless the site sets flow_laya_enabled to a falsy value; unset means on."""
	value = frappe.conf.get("flow_laya_enabled")
	return True if value is None else bool(value)


def _model_path() -> str | None:
	"""The fine-tuned checkpoint to use: the site's, else Flow's own unless the site picked a
	stock Laya model with flow_laya_model."""
	configured = frappe.conf.get("flow_laya_model_path")
	if configured:
		return configured
	return None if frappe.conf.get("flow_laya_model") else DEFAULT_MODEL_PATH


def _threshold() -> float:
	return float(frappe.conf.get("flow_laya_threshold") or DEFAULT_THRESHOLD)


def prefetch() -> None:
	"""Download the ONNX checkpoint into this host's Hugging Face cache, so a worker's first Laya
	message only has to open it (about a second) instead of waiting on the download. Run in the
	background after a migrate; never raises."""
	try:
		path = _model_path()
		if not enabled() or not path or os.path.isdir(path):
			return
		from flow.lib.laya_onnx import _resolve

		_resolve(path, _hf_token())
	except Exception:
		frappe.logger("flow.laya", allow_site=True).warning("Laya checkpoint prefetch failed", exc_info=True)


def enqueue_prefetch() -> None:
	"""after_migrate hook: fetch the checkpoint in a background job, since it is hundreds of MB."""
	if enabled():
		frappe.enqueue("flow.lib.laya_router.prefetch", queue="long", timeout=1800)


def route(text: str, available_tools: set[str]) -> tuple[str, dict[str, Any]] | None:
	"""(tool name, arguments) to run for `text` without the chat model, or None to let the model
	handle it. Never raises: any failure is logged and treated as a fallback."""
	if not enabled() or not isinstance(text, str) or not text.strip():
		return None
	decision = RouteDecision()
	started = time.monotonic()
	try:
		_classify(text, decision)
		_decide(text, decision, available_tools)
	except ModelLoading as e:
		decision.reason = str(e)
	except Exception as e:
		decision.outcome, decision.error = "Error", f"{type(e).__name__}: {e}"[:1000]
		decision.reason = "Laya unavailable or failed; using the chat model"
	decision.latency_ms = int((time.monotonic() - started) * 1000)
	_log(text, decision)
	if decision.outcome == "Routed" and decision.tool:
		return decision.tool, decision.arguments or {}
	return None


def _classify(text: str, decision: RouteDecision) -> None:
	finetuned = _model_path()
	if finetuned:
		result = _finetuned_agent(finetuned).system_one(text, FINETUNED_QUESTIONS)
		result.setdefault("routing", {"model": finetuned})
	else:
		model = frappe.conf.get("flow_laya_model") or None
		result = _router().predict(text, QUESTIONS, model=model)
	answers = result.get("answers") or {}
	intent = answers.get("intent") or {}
	decision.intent = intent.get("choice")
	# answer_confidence is Laya's calibrated confidence, comparable across question types.
	decision.confidence = float(intent.get("answer_confidence") or intent.get("confidence") or 0)
	decision.distribution = {k: float(v) for k, v in (intent.get("probabilities") or {}).items()}
	decision.asks_count = _number(answers.get("asks_count"), "noul")
	decision.complexity = _number(answers.get("complexity"), "score")
	decision.model = str((result.get("routing") or {}).get("model") or "")[:140] or None


def _decide(text: str, decision: RouteDecision, available: set[str]) -> None:
	"""Pick a read-only Flow tool for a confident, simple request; otherwise fall back."""
	from flow.tools.builtins import _find_doctype_in_text, _parse_create_request

	threshold = _threshold()
	if decision.confidence < threshold:
		decision.reason = f"confidence {decision.confidence:.2f} below {threshold:.2f}"
		return

	intent = decision.intent
	tool: str | None = None
	arguments: dict[str, Any] | None = None
	if intent == "TROUBLESHOOTING":
		tool, arguments = "error_diagnosis", {"error": text}
	elif intent == "SMALL_TALK":
		tool, arguments = "small_talk", {"kind": "greeting"}
	elif intent == "HOWTO":
		# HOWTO covers any "how do I ..." but creation_steps only answers how to create a record,
		# so route only when the message asks to create the DocType it names.
		doctype = _creation_target(text)
		if doctype:
			tool, arguments = "creation_steps", {"doctype": doctype}
		else:
			from flow.tools.answers import loose_assignment, loose_status

			arguments = loose_assignment(text)
			tool = "assignment_steps" if arguments else None
			if not arguments:
				arguments = loose_status(text)
				tool = "status_steps" if arguments else None
	elif intent == "RECORD_LOOKUP" and (decision.asks_count or 0) >= threshold:
		from flow.tools.answers import count_arguments

		# The same reading as the exact route; a breakdown or scope it can't map to a field goes
		# to the chat model rather than becoming a plain total.
		arguments = count_arguments(text)
		tool = "count" if arguments else None
	elif intent == "CREATE":
		# Laya never routes to create itself: only the read-only "what values do you need"
		# reply for a request that names a DocType but gives no values.
		parsed = _parse_create_request(text)
		if parsed and not parsed[1]:
			tool, arguments = "required_values", {"doctype": parsed[0]}

	if not tool:
		decision.reason = f"{intent}: no direct Flow path for this request"
		return
	if tool not in available:
		decision.reason = f"{intent}: tool {tool} isn't enabled for this agent"
		return
	decision.outcome, decision.tool, decision.arguments = "Routed", tool, arguments
	decision.reason = f"{intent} at {decision.confidence:.2f}"


CREATE_WORDS = {
	"create", "creating", "add", "adding", "make", "making", "new", "register", "registering",
	"enter", "entering", "raise", "raising", "record", "recording", "set", "setting",
}
FILLER_WORDS = {"a", "an", "the", "new", "one", "another", "up"}
# "assign roles to employees", "link a contact to a customer": acting on records, not creating them.
ASSIGN_WORDS = re.compile(
	r"\b(assign|assigning|link|linking|attach|attaching|grant|granting|allocate|allocating|map|mapping|connect|connecting)\b",
	re.IGNORECASE,
)


def _creation_target(text: str) -> str | None:
	"""The DocType a how-to message asks to create ("guide me through adding a supplier" ->
	Supplier), or None when it asks something else about it ("how to assign roles to employees",
	"how do I add a role to an employee")."""
	from flow.tools.builtins import _doctype_from_phrase

	if ASSIGN_WORDS.search(text or ""):
		return None
	words = re.findall(r"[a-z][a-z\-]*", (text or "").lower())
	for i, word in enumerate(words):
		if word not in CREATE_WORDS or (word in ("set", "setting") and words[i + 1 : i + 2] != ["up"]):
			continue
		start = i + 1
		while start < len(words) and words[start] in FILLER_WORDS:
			start += 1
		for size in range(min(4, len(words) - start), 0, -1):
			doctype = _doctype_from_phrase(" ".join(words[start : start + size]))
			if doctype:
				# "add a role to an employee" adds to another record rather than creating one.
				return None if words[start + size : start + size + 1] in (["to"], ["into"]) else doctype
	return None


def _number(answer: dict[str, Any] | None, key: str) -> float | None:
	if not answer or answer.get(key) is None:
		return None
	return float(answer[key])


# The ONNX model loads in a background thread: the first download (1.6 GB for the fine-tuned
# export) and load can take minutes, far past a web request's timeout, so a request never waits
# for it; until it is ready, messages fall back to the chat model. PyTorch models (the `laya`
# package) load in the request instead, as before: loading PyTorch in a background thread while
# the web server keeps serving requests intermittently crashed the whole process (SIGSEGV).
LOAD_RETRY_SECONDS = 300
_models: dict[str, Any] = {}
_loading: dict[str, dict[str, Any]] = {}
_router_lock = threading.Lock()


class ModelLoading(Exception):
	"""The Laya model isn't loaded yet; this message goes to the chat model."""


def _router() -> Any:
	"""Laya's Router, loaded once per worker (it downloads a checkpoint on first use)."""

	def load():
		from laya import Router  # optional dependency: ImportError means "unavailable"

		return Router()

	return _model("router", load, background=False)


def _finetuned_agent(path: str) -> Any:
	"""The Flow fine-tuned checkpoint at `path`, loaded once per worker: on Flow's ONNX runner
	when it ships laya.onnx (or flow_laya_backend says so), else with the `laya` package."""
	# Site config is read here, in the request: the loading thread has no Frappe context.
	backend, token = (frappe.conf.get("flow_laya_backend") or "").lower(), _hf_token()
	key = f"finetuned:{path}"
	if key not in _backends:
		_backends[key] = _use_onnx(path, backend, token)
	onnx = _backends[key]

	def load():
		if onnx:
			from flow.lib.laya_onnx import OnnxLaya

			return OnnxLaya(path, token=token)
		import laya  # optional dependency (pulls in PyTorch)

		return laya.load(path)

	return _model(key, load, background=onnx)


_backends: dict[str, bool] = {}


def _model(key: str, load, background: bool = True) -> Any:
	"""The loaded model for `key`. In the background: ModelLoading after starting (or while
	running) its load in a thread; a failed load is reported, then retried after
	LOAD_RETRY_SECONDS. Otherwise loaded in this request, once per worker."""
	if key in _models:
		return _models[key]
	if not background:
		with _router_lock:
			if key not in _models:
				_models[key] = load()
		return _models[key]
	with _router_lock:
		state = _loading.get(key)
		if state is None or (state["error"] and time.monotonic() - state["at"] > LOAD_RETRY_SECONDS):
			state = _loading[key] = {"error": None, "at": time.monotonic()}

			def run():
				try:
					_models[key] = load()
				except Exception as e:
					state["error"], state["at"] = f"{type(e).__name__}: {e}"[:900], time.monotonic()
					logging.getLogger("flow.laya").warning("Laya model %s failed to load: %s", key, state["error"])

			threading.Thread(target=run, name=f"flow-laya-load-{key}", daemon=True).start()
	if key in _models:
		return _models[key]
	if state["error"]:
		raise RuntimeError(f"Laya model failed to load: {state['error']}")
	raise ModelLoading("Laya model is still loading in the background; the chat model answers meanwhile")


def _hf_token() -> str | None:
	"""Read token for a private Hugging Face checkpoint: the HF_TOKEN environment variable, or the
	flow_hf_token config key for hosts that only offer config, not environment variables (Frappe Cloud)."""
	return os.environ.get("HF_TOKEN") or frappe.conf.get("flow_hf_token") or None


def _use_onnx(path: str, backend: str = "", token: str | None = None) -> bool:
	from flow.lib.laya_onnx import ONNX_FILE, is_onnx_checkpoint

	if backend in ("onnx", "laya"):
		return backend == "onnx"
	if os.path.isdir(path):
		return is_onnx_checkpoint(path)
	# A Hugging Face repo id: look for the exported graph among its files. Bounded, because this
	# runs in the chat request and the call has no timeout of its own; a failure (offline, private
	# repo without a token) propagates so route() logs it, rather than being read as "no ONNX".
	from concurrent.futures import ThreadPoolExecutor

	from huggingface_hub import HfApi

	pool = ThreadPoolExecutor(max_workers=1)
	try:
		files = pool.submit(HfApi().list_repo_files, path, token=token).result(timeout=HF_PROBE_SECONDS)
	finally:
		pool.shutdown(wait=False)
	return ONNX_FILE in files


def _log(text: str, decision: RouteDecision) -> None:
	"""Record the decision as a Flow Route Log row and in the flow.laya log file."""
	logger = frappe.logger("flow.laya", allow_site=True)
	# Frappe loggers keep only warnings (errors in production) by default; every decision is
	# worth a line here, so this one logger records info.
	logger.setLevel(logging.INFO)
	logger.info(
		{
			"outcome": decision.outcome,
			"intent": decision.intent,
			"confidence": decision.confidence,
			"tool": decision.tool,
			"latency_ms": decision.latency_ms,
			"error": decision.error,
		}
	)
	try:
		frappe.get_doc(
			{
				"doctype": "Flow Route Log",
				"run": frappe.flags.flow_run,
				"input": text[:INPUT_LOG_LIMIT],
				"intent": decision.intent,
				"confidence": decision.confidence,
				"distribution": frappe.as_json(decision.distribution),
				"asks_count": decision.asks_count,
				"complexity": decision.complexity,
				"outcome": decision.outcome,
				"tool": decision.tool,
				"reason": decision.reason[:140],
				"model": decision.model,
				"latency_ms": decision.latency_ms,
				"error": decision.error,
			}
		).insert(ignore_permissions=True)
		# The chat stream stays open while the browser model answers, so without this the row
		# only becomes visible (to laya_status and the metrics) when the whole request ends.
		if not frappe.flags.in_test:
			frappe.db.commit()
	except Exception:
		logger.exception("could not write Flow Route Log")


def status() -> dict[str, Any]:
	"""Why Laya is or isn't answering on this site: the settings it reads, whether its checkpoint
	has loaded in this worker, and the last decisions logged. Never includes the token."""
	path = _model_path()
	out: dict[str, Any] = {
		"enabled": enabled(),
		"threshold": _threshold(),
		"model_path": path,
		"token_set": bool(_hf_token()),
		"decisions_logged": frappe.db.count("Flow Route Log"),
	}
	try:
		import onnxruntime

		out["onnxruntime"] = onnxruntime.__version__
	except Exception as e:
		out["onnxruntime"] = f"missing: {type(e).__name__}"
	if path:
		key = f"finetuned:{path}"
		state = _loading.get(key) or {}
		out["onnx_checkpoint"] = _backends.get(key)
		out["loaded_in_this_worker"] = key in _models
		out["loading_error"] = state.get("error")
	last = frappe.get_all(
		"Flow Route Log",
		fields=["creation", "outcome", "intent", "confidence", "reason", "error"],
		order_by="creation desc",
		limit=3,
	)
	out["last_decisions"] = [{**row, "creation": str(row.creation)} for row in last]
	return out


def metrics(days: int = 7) -> dict[str, Any]:
	"""Laya decisions over the last `days`: how many were routed, fell back or errored, the
	fallback rate, average confidence and latency, and the routed count per intent."""
	since = frappe.utils.add_days(frappe.utils.now_datetime(), -int(days))
	rows = frappe.get_all(
		"Flow Route Log",
		filters={"creation": [">=", since]},
		fields=["outcome", "intent", "confidence", "latency_ms"],
		limit_page_length=0,
	)
	total = len(rows)
	count = lambda outcome: sum(1 for r in rows if r.outcome == outcome)  # noqa: E731
	routed_by_intent: dict[str, int] = {}
	for r in rows:
		if r.outcome == "Routed":
			routed_by_intent[r.intent] = routed_by_intent.get(r.intent, 0) + 1
	classified = [r for r in rows if r.outcome != "Error"]
	return {
		"enabled": enabled(),
		"days": int(days),
		"total": total,
		"routed": count("Routed"),
		"fallback": count("Fallback"),
		"errors": count("Error"),
		"fallback_rate": round(count("Fallback") / total, 4) if total else None,
		"error_rate": round(count("Error") / total, 4) if total else None,
		"avg_confidence": round(sum(r.confidence or 0 for r in classified) / len(classified), 4) if classified else None,
		"avg_latency_ms": round(sum(r.latency_ms or 0 for r in rows) / total) if total else None,
		"routed_by_intent": routed_by_intent,
	}
