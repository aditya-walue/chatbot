# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE
#
# Portions (build_head / build_sequence / render_options / the softmax-with-temperature decode)
# follow laya/common.py and laya/onnx_agent.py from https://github.com/NandhaKishorM/laya
# (Apache License 2.0, Copyright the Laya authors). See scripts/laya/LICENSE-laya.

"""Run a Laya checkpoint exported to ONNX without PyTorch or the `laya` package.

The `laya` package imports PyTorch (laya/common.py), which on a Linux server pulls in several GB
of CUDA libraries — impractical on hosts like Frappe Cloud. Flow already depends on onnxruntime
and (through litellm) on tokenizers, numpy and huggingface_hub, so an exported checkpoint can be
run with those alone. This module reproduces exactly what Laya's ONNXAgent does for the subset Flow
uses — single string states, choice / noul / score questions — and returns the same answer shape.

A checkpoint directory (local, or a Hugging Face repo) needs:
    rl_agent_config.json       max_len / head_max_len / per-type temperatures
    tokenizer/tokenizer.json   (+ tokenizer_config.json for the special tokens)
    laya.onnx [+ laya.onnx.data]   from Laya's scripts/export_onnx.py

Tested for parity against laya's own agent in flow/tests/test_laya_onnx.py and
scripts/laya/check_onnx_parity.py.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
TEMP_MIN, TEMP_MAX = 0.5, 5.0
OPTION_MAX_TOKENS = 48
ONNX_FILE = "laya.onnx"


def is_onnx_checkpoint(path: str) -> bool:
	return os.path.isfile(os.path.join(path, ONNX_FILE))


class OnnxLaya:
	"""`system_one(text, questions)` over an exported checkpoint, like laya's agents."""

	def __init__(self, model_id_or_path: str, token: str | None = None, cache_dir: str | None = None):
		import onnxruntime as ort

		self.model_id = model_id_or_path
		model_dir = _resolve(model_id_or_path, token, cache_dir)
		with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
			self.cfg = json.load(f)
		self.tok = _Tokenizer(os.path.join(model_dir, "tokenizer"))
		self.max_len = int(self.cfg.get("max_len", 1024))
		self.head_max_len = int(self.cfg.get("head_max_len", 256))
		self.temperature = [_clamp(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
		self.temperature_by_options = {k: _clamp(v) for k, v in (self.cfg.get("temperature_by_options") or {}).items()}

		options = ort.SessionOptions()
		options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
		self.session = ort.InferenceSession(
			os.path.join(model_dir, ONNX_FILE), options, providers=["CPUExecutionProvider"]
		)
		self._lock = threading.Lock()

	def system_one(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
		internal = {qid: _internal(q) for qid, q in questions.items()}
		state_ids = self.tok(str(state).replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
		items = []
		for qid, q in internal.items():
			ids, markers = _build_sequence(self.tok, q, state_ids, self.max_len, self.head_max_len)
			if len(markers) != len(_render_options(q)):
				raise ValueError(f"question {qid!r}: options don't fit in max_len={self.max_len}")
			items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})

		feed = _collate(items, self.tok.pad_token_id)
		with self._lock:
			logits, act = self.session.run(["logits", "act_logits"], feed)
		answers = {qid: self._decode(logits[r], act[r], items[r], q) for r, (qid, q) in enumerate(internal.items())}
		return {"model": "laya-onnx", "answers": answers, "routing": {"model": self.model_id}}

	def _decode(self, logits: np.ndarray, act: np.ndarray, item: dict, q: dict) -> dict[str, Any]:
		k = len(item["markers"])
		qt = item["qtype"]
		t_scale = self.temperature_by_options.get(_temp_bucket(qt, k), self.temperature[qt])
		z = logits[:k] / t_scale
		p = np.exp(z - z.max())
		p = p / p.sum()
		answer_conf = round(float(np.clip(np.max(p[:k]), 0.0, 1.0)), 4)
		confidence = round(_entropy_confidence(p, k), 4)
		action = {"act_probability": round(float(act[0]), 4)}
		if q["t"] == "choice":
			keys = list(q["crit"].keys())
			return {
				"type": "choice",
				"choice": keys[int(p.argmax())],
				"probabilities": {key: round(float(v), 4) for key, v in zip(keys, p)},
				"confidence": confidence,
				"answer_confidence": answer_conf,
				"action": action,
			}
		if q["t"] == "score":
			return {
				"type": "score",
				"score": round(float((np.arange(k) * p).sum()), 4),
				"probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
				"confidence": confidence,
				"answer_confidence": answer_conf,
				"action": action,
			}
		return {
			"type": "noul",
			"noul": round(float(p[1]), 4),
			"confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
			"answer_confidence": answer_conf,
			"action": action,
		}


# ── checkpoint files ───────────────────────────────────────────────────────────────────────


def _resolve(model_id_or_path: str, token: str | None = None, cache_dir: str | None = None) -> str:
	"""A local checkpoint directory, downloading a Hugging Face repo (only the files needed; a
	private repo needs `token`, HF_TOKEN or the saved login) when given a repo id. `cache_dir`
	replaces the default ~/.cache/huggingface, which hosted benches (Frappe Cloud) can't write to."""
	if os.path.isdir(model_id_or_path):
		return model_id_or_path
	from huggingface_hub import snapshot_download

	return snapshot_download(
		model_id_or_path,
		allow_patterns=["rl_agent_config.json", "tokenizer/*", f"{ONNX_FILE}", f"{ONNX_FILE}.data"],
		token=token or os.environ.get("HF_TOKEN") or None,
		cache_dir=cache_dir,
	)


class _Tokenizer:
	"""The slice of a Hugging Face tokenizer Laya's sequence builder uses, over `tokenizers`."""

	def __init__(self, path: str):
		from tokenizers import Tokenizer

		self._tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
		self._tok.no_truncation()
		self._tok.no_padding()
		config = {}
		config_path = os.path.join(path, "tokenizer_config.json")
		if os.path.exists(config_path):
			with open(config_path) as f:
				config = json.load(f)
		token = lambda name, default: _content(config.get(name)) or default  # noqa: E731
		self.cls_token = token("cls_token", "[CLS]")
		self.sep_token = token("sep_token", "[SEP]")
		self.mask_token = token("mask_token", "[MASK]")
		self.pad_token = token("pad_token", "[PAD]")
		self.cls_token_id = self._tok.token_to_id(self.cls_token)
		self.sep_token_id = self._tok.token_to_id(self.sep_token)
		self.mask_token_id = self._tok.token_to_id(self.mask_token)
		self.pad_token_id = self._tok.token_to_id(self.pad_token) or 0

	def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False, max_length: int | None = None):
		ids = self._tok.encode(text, add_special_tokens=add_special_tokens).ids
		if truncation and max_length is not None:
			ids = ids[:max_length]
		return {"input_ids": ids}


def _content(value: Any) -> str | None:
	if isinstance(value, dict):
		return value.get("content")
	return value


# ── sequence building and decoding (as laya/common.py) ──────────────────────────────────────


def _internal(q: dict[str, Any]) -> dict[str, Any]:
	return {"t": q["type"], "ins": q.get("instructions", ""), "crit": q.get("criteria")}


def _render_criterion(value: Any) -> str:
	if isinstance(value, str):
		return value
	return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _render_options(q: dict[str, Any]) -> list[str]:
	t, crit = q["t"], q.get("crit")
	if t == "choice":
		return [str(k) if v is None or v == "" else "%s: %s" % (k, _render_criterion(v)) for k, v in crit.items()]
	if t == "score":
		return ["level %d: %s" % (i, _render_criterion(c)) for i, c in enumerate(crit)]
	crit = crit or {}
	false_crit, true_crit = crit.get("false"), crit.get("true")
	# Laya's default noul labels are "false" / "true" (laya/common.py _DEFAULT_NOUL_LABELS).
	return [
		"false: " + (_render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
		"true: " + (_render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
	]


def _build_head(tok: _Tokenizer, q: dict[str, Any], head_max_len: int) -> tuple[list[int], list[int]]:
	mask = tok.mask_token
	opts = _render_options(q)
	ins = str(q["ins"]).replace(mask, " ")
	head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
	opt_ids = [
		[tok.mask_token_id]
		+ tok(" " + o.replace(mask, " "), add_special_tokens=False, truncation=True, max_length=OPTION_MAX_TOKENS)["input_ids"]
		for o in opts
	]
	opt_budget = head_max_len - sum(len(o) for o in opt_ids)
	if opt_budget < 16:
		per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
		opt_ids = [o[:per] for o in opt_ids]
		opt_budget = head_max_len - sum(len(o) for o in opt_ids)
	head_ids = head_ids[: max(8, opt_budget)]
	ids = [tok.cls_token_id, *head_ids, tok.sep_token_id]
	markers = []
	for o in opt_ids:
		markers.append(len(ids))
		ids.extend(o)
	ids.append(tok.sep_token_id)
	return ids, markers


def _build_sequence(tok, q, state_ids, max_len, head_max_len):
	ids, markers = _build_head(tok, q, head_max_len)
	room = max(0, max_len - len(ids) - 1)
	ids = [*ids, *state_ids[:room], tok.sep_token_id]
	return ids[:max_len], [m for m in markers if m < max_len]


def _collate(items: list[dict], pad_id: int) -> dict[str, np.ndarray]:
	n, length = len(items), max(len(it["ids"]) for it in items)
	kmax = max(len(it["markers"]) for it in items)
	ids = np.full((n, length), pad_id, dtype=np.int64)
	attention = np.zeros((n, length), dtype=np.int64)
	marker_pos = np.zeros((n, kmax), dtype=np.int64)
	marker_mask = np.zeros((n, kmax), dtype=bool)
	for i, it in enumerate(items):
		ids[i, : len(it["ids"])] = it["ids"]
		attention[i, : len(it["ids"])] = 1
		k = len(it["markers"])
		marker_pos[i, :k] = it["markers"]
		marker_mask[i, :k] = True
	return {
		"input_ids": ids,
		"attention_mask": attention,
		"marker_pos": marker_pos,
		"marker_mask": marker_mask,
		"qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
	}


def _temp_bucket(qtype: int, k: int) -> str:
	size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
	return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def _clamp(t: Any) -> float:
	return float(min(TEMP_MAX, max(TEMP_MIN, float(t))))


def _entropy_confidence(p: np.ndarray, k: int) -> float:
	if k < 2:
		return 1.0
	p = p[:k]
	entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
	return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))
