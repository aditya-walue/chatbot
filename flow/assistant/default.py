# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""The default chat setup: a small model that runs in the user's browser (WebLLM) and a lean
assistant agent written for it. Both are created and kept up to date from code on install and
migrate, so the panel works with nothing to configure and offers no model or agent choice.
"""

from __future__ import annotations

import frappe

DEFAULT_MODEL_TITLE = "Qwen 2.5 3B (Browser)"
DEFAULT_MODEL_ID = "webllm/Qwen2.5-3B-Instruct-q4f16_1-MLC"

DEFAULT_AGENT_TITLE = "Flow Lite"
DEFAULT_AGENT_MAX_ITERATIONS = 8
DEFAULT_AGENT_TOOLS = (
	"find_doctypes",
	"describe",
	"count",
	"read",
	"create",
	"update",
	"creation_steps",
	"error_diagnosis",
	"required_values",
	"small_talk",
	"show_records",
	"explain_doctype",
	"document_flow",
	"assignment_steps",
	"status_steps",
	"role_permissions",
	"frappe_docs",
	"docs_howto",
	"web_search",
)
DEFAULT_AGENT_INSTRUCTIONS = """You answer questions about the user's ERPNext / Frappe data using tools.

RULES
1. Use only data from tool results or the user. Never invent names, IDs or values.
2. Data questions: show_records (show records, lists, details), count (how many), read (fetch specific field values for reasoning).
3. "How do I assign/link X to Y" -> assignment_steps. "How do I mark/make X paid/completed/..." -> status_steps.
4. "What is X" -> explain_doctype. "What happens after X" -> document_flow. Role permissions -> role_permissions. Errors -> error_diagnosis.
5. To create or change a record the user asked for, call create / update with the values they gave (call describe first if you need field names); if values are missing, call required_values. The user approves before anything is saved.
6. If a tool returns an error, fix that exact problem or tell the user briefly.
7. If no other tool can answer (a general "how does X work" or "how do I ..." about ERPNext / Frappe), call frappe_docs and answer from the page it returns, with its link. Only if it finds nothing, call web_search and list the source links.

Reply in short, plain sentences."""


def sync_default_assistant() -> None:
	"""Ensure the browser model and the default assistant exist, are enabled and match the code.
	Runs after install and migrate."""
	from flow.tools.builtins import sync_builtin_tools

	sync_builtin_tools()
	model = _sync_default_model()
	_sync_default_agent(model)


def default_agent() -> str | None:
	"""The agent every panel chat uses, if it is set up and enabled."""
	if frappe.db.get_value("Flow Agent", DEFAULT_AGENT_TITLE, "enabled"):
		return DEFAULT_AGENT_TITLE
	return None


def _sync_default_model() -> str:
	name = frappe.db.get_value("Flow Model", {"model_id": DEFAULT_MODEL_ID}, "name")
	if not name:
		doc = frappe.get_doc(
			{"doctype": "Flow Model", "title": DEFAULT_MODEL_TITLE, "model_id": DEFAULT_MODEL_ID, "enabled": 1}
		).insert(ignore_permissions=True)
		return doc.name
	if not frappe.db.get_value("Flow Model", name, "enabled"):
		frappe.db.set_value("Flow Model", name, "enabled", 1)
	return name


def _sync_default_agent(model: str) -> None:
	values = {
		"model": model,
		"enabled": 1,
		"is_system_generated": 1,
		"max_iterations": DEFAULT_AGENT_MAX_ITERATIONS,
		"instructions": DEFAULT_AGENT_INSTRUCTIONS,
	}
	if frappe.db.exists("Flow Agent", DEFAULT_AGENT_TITLE):
		doc = frappe.get_doc("Flow Agent", DEFAULT_AGENT_TITLE)
		doc.update(values)
	else:
		doc = frappe.get_doc({"doctype": "Flow Agent", "title": DEFAULT_AGENT_TITLE, **values})
	doc.set("tools", [{"tool": slug} for slug in DEFAULT_AGENT_TOOLS if frappe.db.exists("Flow Tool", slug)])
	doc.save(ignore_permissions=True) if not doc.is_new() else doc.insert(ignore_permissions=True)
