# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Read-only tools that answer common questions completely from site data, with routes that run
them without the chat model:

- show_records    a record by ID ("show the details of <ID>", "status of <ID>"), or a list
                  ("list all sales invoices", "show open todos", "which employees do we have")
- explain_doctype "what is a <doctype>?" — module, purpose, lifecycle, main fields, links
- document_flow   "what happens after a <doctype> is submitted?" — lifecycle, workflow, and the
                  documents ERPNext connects before and after it
- assignment_steps "how do I assign <doctype> to <doctype>?" — where one record type is attached
                  to another (a field, a table, or through a linked record), from the metadata
- status_steps    "how do I mark <doctype> as <status>?" — how a record reaches a status: the
                  connected document that sets it (Paid -> Payment Entry), or submit / cancel

Every answer is built from the live metadata and records the user may read; a route only fires
when it recognises a real DocType or record, otherwise the message goes on to Laya / the model.
"""

from __future__ import annotations

import re
from typing import Annotated, Any
from urllib.parse import quote, urlparse

import frappe
from frappe import _
from frappe.utils import get_url_to_form, get_url_to_list, get_url_to_workspace

from flow.lib.tool import LayaRoute, tool

# builtins imports this module to register its tools, so its helpers are imported inside the
# functions below rather than at the top (which would be circular).
DocTypeArg = Annotated[str, 'Exact DocType name, e.g. "Sales Invoice", "Customer", "Item".']
FiltersArg = Annotated[
	dict[str, Any] | None,
	'Field equality or range conditions as a JSON object, e.g. {"status": "Paid"}. Omit for all records.',
]


def _builtins():
	from flow.tools import builtins

	return builtins

LIST_LIMIT = 20
MAX_WORKSPACES = 3
MAX_ACCESS_ROWS = 5000  # most rows read when matching a name to roles or users
MAX_STOCK_ROWS = 5000  # most reorder levels / stock records low_stock reads
MAX_RECORD_TYPE_WORDS = 4  # words of a record type's name
MAX_UNAPPLIED_WORDS = 3  # leftover words of a list request that may be passed over
# The most records one list shows, whatever limit is asked for.
LIST_ALL_LIMIT = 200
MAIN_FIELDS_LIMIT = 6

# A token that looks like a record ID: letters and digits joined by dashes/dots/slashes, with a
# digit somewhere ("ACC-SINV-2026-00008", "AB-XYZ-00003", "PO/24/001").
RECORD_ID = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-./][A-Za-z0-9]+)*[-./]?\d[A-Za-z0-9-./]*\b")
SHOW_RECORD = re.compile(
	r"\b(show|details?|detail|info(?:rmation)?|status|open|view|display|get|tell\s+me\s+about|what\s+is)\b",
	re.IGNORECASE,
)
# Asking to change a record rather than see it: those go to the model (and the approval flow).
WRITE_INTENT = re.compile(r"\b(create|add|make|update|change|set|edit|delete|remove|cancel|submit|amend|rename|close|mark)\b", re.IGNORECASE)
LIST_REQUEST = re.compile(
	r"^\s*(?:please\s+)?(?:list|show(?:\s+me)?|get(?:\s+me)?|display|which|what|any|"
	r"(?:is|are)\s+there(?:\s+any)?|do\s+(?:we|i)\s+have(?:\s+any)?)\s+"
	r"(?:(?:all|the|my|our|every)\s+)*(?:(?P<order>latest|recent|newest|last|oldest)\s+)?"
	r"(?P<rest>[a-z][a-z \-]*?)"
	r"(?:\s+(?:do\s+we\s+have|are\s+there|exist|in\s+(?:the\s+)?system|we\s+have|present))?\s*[?.!]*\s*$",
	re.IGNORECASE,
)
DEFINITION = re.compile(
	r"^\s*(?:what\s+(?:is|are|does)\s+(?:a|an|the)?\s*|explain\s+(?:what\s+(?:a|an)\s+)?|define\s+|meaning\s+of\s+)"
	r"(?P<what>[a-z][a-z \-]*?)(?:\s+(?:mean|means|used\s+for|for|do|in\s+erpnext))?\s*[?.!]*\s*$",
	re.IGNORECASE,
)
FLOW_QUESTION = re.compile(
	r"^\s*(?:what\s+(?:happens|comes|is\s+next|follows)\s+(?:after|once|when)\s+(?:a|an|the)?\s*"
	r"|what\s+comes\s+after\s+(?:a|an|the)?\s*|next\s+steps?\s+(?:after|for)\s+(?:a|an|the)?\s*)"
	r"(?P<what>[a-z][a-z \-]*?)(?:\s+(?:is|gets|has\s+been)\s+(?:submitted|created|approved|saved|made))?\s*[?.!]*\s*$",
	re.IGNORECASE,
)


# ── show_records ──────────────────────────────────────────────────────────────────────────


def _route_show_records(text: str) -> dict[str, Any] | None:
	# A real record ID in a message that isn't asking to change anything is a request to see
	# it, however it is phrased ("pull up", "can you get", "about").
	by_id = _record_from_text(text) if not WRITE_INTENT.search(text or "") else None
	if by_id:
		arguments: dict[str, Any] = {"doctype": by_id[0], "names": [by_id[1]]}
		about = asked_fields(by_id[0], text, exclude=by_id[1], record=by_id[1])
		if about:
			arguments["about"] = about
		return arguments
	match = LIST_REQUEST.match(text or "")
	if not match:
		return None
	return _list_arguments(match.group("rest"), match.group("order"))


def _record_from_text(text: str) -> tuple[str, str] | None:
	"""(DocType, ID) for a record ID in the text that exists and the user may read. The DocType
	comes from the text when named, else from the naming series the ID's prefix belongs to."""
	named = _builtins()._find_doctype_in_text(text)
	tokens = RECORD_ID.findall(text or "")
	if named:
		# With the DocType named, a word may be its ID even outside a naming series: random IDs
		# ("1bj14mc8ae", or all letters like "hfpssoeeph"). Digit-bearing words are tried first;
		# a word only matches when a record with exactly that ID exists.
		words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-./]{3,}", text or "")
		# An all-letter word that is part of a DocType's name ("employee" in "assign a role to
		# new employee") is a record type, not a random ID.
		type_words = {w for name in _builtins()._doctype_names() for w in name.split()}
		tokens += [w for w in words if re.search(r"\d", w)] + [
			w for w in words if len(w) >= 8 and w.isalpha() and w.lower() not in type_words
		]
	for token in dict.fromkeys(tokens):
		token = token.strip(".-/")
		for doctype in ([named] if named else []) + _doctypes_for_series(token):
			if frappe.db.exists(doctype, {"name": token}) and frappe.has_permission(doctype, "read", token):
				return doctype, token
	return None


def _doctypes_for_series(record_id: str) -> list[str]:
	"""DocTypes whose naming series options start like this ID ("ACC-SINV-2026-00008" -> the
	DocTypes with an "ACC-SINV-" series), from the DocType and any Property Setter override."""
	prefix = re.match(r"^(.*?[-./])\d", record_id)
	if not prefix or len(prefix.group(1)) < 3:
		return []
	stem = re.split(r"[-./]\d", record_id)[0].split(".")[0]
	cache = frappe.flags.flow_series_doctypes = getattr(frappe.flags, "flow_series_doctypes", None) or {}
	if stem in cache:
		return cache[stem]
	like = ["like", f"%{stem}%"]
	doctypes = frappe.get_all("DocField", filters={"fieldname": "naming_series", "options": like}, pluck="parent")
	doctypes += frappe.get_all(
		"Property Setter",
		filters={"field_name": "naming_series", "property": "options", "value": like},
		pluck="doc_type",
	)
	res = list(dict.fromkeys(doctypes))
	cache[stem] = res
	return res


def _list_arguments(rest: str, order: str | None) -> dict[str, Any] | None:
	"""Arguments for a list request: the DocType, plus a status filter when a word in the request
	is one of its status values ("open todos", "unpaid customer invoices")."""
	arguments = list_target(rest) or _embedded_target(rest)
	if not arguments:
		return None
	if order in ("oldest",):
		arguments["order_by"] = "creation asc"
	return arguments


def _embedded_target(phrase: str) -> dict[str, Any] | None:
	"""A record type followed by a few more words ("holiday list that is added"): the list of it,
	with the trailing words handed on as `unapplied`, so the answer says plainly what was not used
	as a filter. Words before the record type ("broken invoices", "is a todo") are qualifiers and
	stay with the model, and so does a long tail, which is no filler."""
	words = (phrase or "").strip().split()
	for size in range(min(MAX_RECORD_TYPE_WORDS, len(words)), 0, -1):
		tail = words[size:]
		doctype = _record_type(words[:size])
		if doctype and len(tail) <= MAX_UNAPPLIED_WORDS:
			return {"doctype": doctype, "unapplied": " ".join(tail)} if tail else {"doctype": doctype}
	return None


def list_target(phrase: str, scope: set[str] | None = None) -> dict[str, Any] | None:
	"""{"doctype"[, "filters"]} for a phrase naming records, optionally with one status word:
	"sales invoices", "unpaid sales invoices", "unpaid customer invoices", "overdue invoices".
	None when a word is neither part of the record type nor one of its statuses ("broken
	invoices"), so the chat model interprets it."""
	words = (phrase or "").strip().split()
	if not words:
		return None
	doctype = _record_type(words, scope=scope)
	if doctype:
		return {"doctype": doctype}
	for i, word in enumerate(words):
		rest = words[:i] + words[i + 1 :]
		doctype = rest and _record_type(rest, status_word=word, scope=scope)
		status = doctype and _status_value(doctype, word)
		if status:
			return {"doctype": doctype, "filters": {"status": status}}
	return None


def _record_type(words: list[str], status_word: str | None = None, scope: set[str] | None = None) -> str | None:
	"""The DocType `words` name: exactly ("sales invoices"), or by the last word as the end of
	DocType names, narrowed by the other words ("customer invoices" -> the Invoice type with a
	Customer link), by `status_word` being one of its statuses and by `scope` (the modules of an
	app or module the user named: "leads in Frappe CRM" -> CRM Lead). Of several that still fit,
	the one with the most matching records, as the one the site actually uses; an exact name
	("Lead") also competes with DocTypes ending in it ("CRM Lead") that way."""
	builtins = _builtins()
	exact = builtins._doctype_from_phrase(" ".join(words), fuzzy=True)
	if exact:
		siblings = [
			name for key, name in builtins._doctype_names().items()
			if key.endswith(" " + exact.lower()) and frappe.has_permission(name, "read")
		]
		if not siblings and not scope:
			return exact
		candidates = [exact, *siblings]
		if scope:
			candidates = [c for c in candidates if _module_key(c) in scope]
		return _most_used(candidates, status_word, prefer=exact)
	noun = _singular(words[-1].lower())
	candidates = [
		name for key, name in builtins._doctype_names().items()
		if key.split()[-1] == noun and frappe.has_permission(name, "read")
	]
	if scope:
		candidates = [c for c in candidates if _module_key(c) in scope]
	for word in words[:-1]:
		named = builtins._doctype_from_phrase(word)
		narrowed = [c for c in candidates if word.lower() in c.lower().split() or (named and _links_to(c, named))]
		if not narrowed:
			return None  # a qualifier none of them fits
		candidates = narrowed
	return _most_used(candidates, status_word)


def _most_used(candidates: list[str], status_word: str | None, prefer: str | None = None) -> str | None:
	"""Of `candidates`, those with `status_word` as a status (when given), then the one with the
	most matching records, then the most records; ties go to `prefer`."""
	if status_word:
		candidates = [c for c in candidates if _status_value(c, status_word)]
	if len(candidates) <= 1:
		return candidates[0] if candidates else None

	def used(doctype: str) -> tuple[int, int, int]:
		status = status_word and _status_value(doctype, status_word)
		return (
			frappe.db.count(doctype, {"status": status}) if status else 0,
			frappe.db.count(doctype),
			1 if doctype == prefer else 0,
		)

	return max(candidates, key=used)


def _module_key(doctype: str) -> str:
	return (frappe.get_meta(doctype).module or "").lower()


def _scope(words: str) -> set[str] | None:
	"""Modules (lower-cased) an "in <words>" phrase names: a module ("FCRM", "Accounts") or an
	installed app by name or title ("crm", "Frappe CRM", "HR"); None when it names neither
	("in the system")."""
	wanted = " ".join(words.lower().split())
	if not wanted:
		return None
	modules = getattr(frappe.flags, "flow_all_modules", None)
	if modules is None:
		modules = frappe.flags.flow_all_modules = {m.lower(): m for m in frappe.get_all("Module Def", pluck="name")}
	if wanted in modules:
		return {wanted}
	for app in frappe.get_installed_apps():
		title = (frappe.get_hooks("app_title", app_name=app) or [app])[0].lower()
		if wanted in (app.lower(), title, title.removeprefix("frappe ").strip()):
			return {m.lower() for m in frappe.get_all("Module Def", filters={"app_name": app}, pluck="name")}
	return None


# "how many leads", "what is the total leads? stage wise", "number of open todos by priority",
# "how many leads in Frappe CRM", "total deals per stage"
COUNT_PREFIX = re.compile(
	r"^\s*(?:please\s+)?(?:how\s+many|count(?:\s+of)?(?:\s+the)?|(?:the\s+)?number\s+of|"
	r"what\s+(?:is|are)\s+the\s+(?:total\s+)?(?:number|count)\s+of|what\s+(?:is|are)\s+the\s+total|"
	r"(?:give\s+me\s+)?(?:the\s+)?total(?:\s+number)?(?:\s+of)?)\s+(?:total\s+)?(?:(?:the|all)\s+)?",
	re.IGNORECASE,
)
GROUP_WORDS = re.compile(
	r"\b(?:(?P<a>[a-z]+)[\s-]?wise|(?:grouped\s+|broken\s+down\s+|breakdown\s+)?by\s+(?:the\s+)?(?P<b>[a-z]+)"
	r"|per\s+(?P<c>[a-z]+)|for\s+each\s+(?P<d>[a-z]+))\b",
	re.IGNORECASE,
)
COUNT_TAIL = re.compile(
	r"\s+(?:(?:are\s+there|is\s+there|do\s+(?:we|i)\s+have|we\s+have|exist|exists|present)"
	r"(?:\s+(?:in|on)\s+(?:the\s+)?(?:system|site))?|"
	r"(?:are|is)\s+(?:in|on)\s+(?:the\s+)?(?:system|site)|in\s+(?:the\s+)?(?:system|site))$",
	re.IGNORECASE,
)
def count_arguments(text: str) -> dict[str, Any] | None:
	"""count() arguments for a count question, or None."""
	clean = re.sub(r"[?!.,;]+", " ", text or "")
	match = COUNT_PREFIX.match(clean)
	if not match:
		return None
	rest = clean[match.end() :]
	group = GROUP_WORDS.search(rest)
	group_word = None
	if group:
		group_word = next(g for g in group.groups() if g)
		rest = rest[: group.start()] + rest[group.end() :]
	rest = " ".join(rest.split())
	rest = COUNT_TAIL.sub("", rest).strip()
	scope = None
	in_match = re.search(r"\s+(?:in|from|on)\s+(?:the\s+)?(?P<scope>[a-z][a-z ]*)$", rest, re.IGNORECASE)
	if in_match:
		scope = _scope(in_match.group("scope"))
		if scope:
			rest = rest[: in_match.start()]
	arguments = list_target(rest, scope=scope)
	if not arguments:
		return None
	if group_word:
		field = _group_field(arguments["doctype"], group_word)
		if not field:
			return None  # an unknown breakdown: let the model interpret it
		arguments["group_by"] = field
	arguments.pop("order_by", None)
	return arguments


def _group_field(doctype: str, word: str) -> str | None:
	"""The field "<word> wise" / "by <word>" names, by its fieldname or label. None otherwise:
	the chat model then picks the field from the DocType's own fields (count's group_by)."""
	meta = frappe.get_meta(doctype)
	word = _singular(word.lower())
	for f in meta.fields:
		if f.fieldtype in ("Select", "Link", "Data", "Check", "Date") and word in (
			_singular(f.fieldname.lower()),
			_singular((f.label or "").lower()),
		):
			return f.fieldname
	return None


def _singular(word: str) -> str:
	if word.endswith("ies"):
		return word[:-3] + "y"
	if word.endswith(("ses", "xes", "ches", "shes")):
		return word[:-2]
	return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _links_to(doctype: str, other: str) -> bool:
	return any(f.fieldtype == "Link" and f.options == other for f in frappe.get_meta(doctype).fields)


def _status_value(doctype: str, word: str) -> str | None:
	field = frappe.get_meta(doctype).get_field("status")
	if not field or field.fieldtype != "Select" or not field.options:
		return None
	options = {o.strip().lower(): o.strip() for o in field.options.split("\n") if o.strip()}
	return options.get(word.lower())


@tool(final_answer=True, route=_route_show_records)
def show_records(
	doctype: DocTypeArg,
	names: Annotated[list[str] | None, "Record IDs; omit for a recent list."] = None,
	filters: FiltersArg = None,
	order_by: Annotated[str | None, 'e.g. "creation desc".'] = None,
	about: Annotated[list[str] | None, 'Fieldnames the user asked about for one record, e.g. ["total_taxes_and_charges", "taxes"].'] = None,
	limit: Annotated[int | None, f"How many records to list, up to {LIST_ALL_LIMIT}. Default 20; when the user wants all of them, pass the total count."] = None,
	start: Annotated[int, "Offset for pagination (number of records to skip). Default 0."] = 0,
	unapplied: Annotated[str | None, "Words of the request that could not be used as a filter; the answer says so."] = None,
) -> dict[str, Any]:
	"""Show records to the user: one by ID (plus the fields asked about) or a recent or filtered
	list. Use for show, view, list, "is there any <record type>" and "what is the <field> of
	<record>"."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	route = doctype.lower().replace(" ", "-")
	if frappe.get_meta(doctype).issingle:
		return {"answer": f"**{doctype}** is a single settings page — [open it](/desk/{route}).", "doctype": doctype}
	columns = _builtins()._key_columns(doctype)
	filters = _tidy_filters(doctype, filters)
	conditions = dict(filters or {})
	if names:
		conditions["name"] = ["in", names]
	limit_val = min(int(limit), LIST_ALL_LIMIT) if limit else LIST_LIMIT
	start_val = max(int(start or 0), 0)
	rows = frappe.get_list(
		doctype, filters=conditions, fields=columns, order_by=order_by or "creation desc",
		limit=limit_val,
		start=start_val,
	)
	if names:
		total = len(rows)
	elif start_val == 0 and len(rows) < limit_val:
		total = len(rows)
	else:
		total = frappe.db.count(doctype, conditions)
	meta = frappe.get_meta(doctype)
	labels = {c: (meta.get_field(c).label if meta.get_field(c) else c.replace("_", " ").title()) for c in columns}

	passed_over = f"\n\nNot used as a filter: \"{unapplied}\"." if unapplied else ""
	if not rows:
		what = f"{doctype} {', '.join(names)}" if names else f"{doctype} records" + (" matching that" if filters else "")
		return {"answer": f"No {what} found (or you don't have access to them).{passed_over}", "doctype": doctype}

	if len(rows) == 1 and names:
		row = rows[0]
		header = f"**[{row.name}](/desk/{route}/{quote(str(row.name))})** — {doctype}"
		asked_about = [f for f in (about or []) if meta.get_field(f)]
		if asked_about:
			# A question about some fields gets just those fields ("what is the tax component?").
			asked = _asked_values(doctype, row.name, asked_about)
			if not asked:
				labels_asked = ", ".join(meta.get_field(f).label for f in asked_about[:3])
				asked = [f"**{labels_asked}** {'is' if len(asked_about) == 1 else 'are'} not filled in on this record."]
			return {"answer": "\n".join([header, "", *asked]), "doctype": doctype, "names": [row.name]}
		lines = [header]
		lines += [f"- **{labels[c]}**: {_fmt(row.get(c))}" for c in columns if c != "name" and row.get(c) not in (None, "")]
		return {"answer": "\n".join(lines), "doctype": doctype, "names": [row.name]}

	shown = [c for c in columns if c != "name"][:4]
	header = "| ID | " + " | ".join(labels[c] for c in shown) + " |"
	sep = "|---" * (len(shown) + 1) + "|"
	body = [
		f"| [{r.name}](/desk/{route}/{quote(str(r.name))}) | " + " | ".join(_fmt(r.get(c)) for c in shown) + " |" for r in rows
	]
	qualifier = " ".join(f"{k} = {v}" for k, v in (filters or {}).items())
	title = f"**{total} {doctype} record{'s' if total != 1 else ''}**" + (f" ({qualifier})" if qualifier else "")
	if start_val > 0:
		start_idx = start_val + 1
		end_idx = start_val + len(rows)
		more = f"\n\nShowing records {start_idx} to {end_idx} of {total}."
	elif total > len(rows):
		more = f"\n\nShowing the latest {len(rows)}."
	else:
		more = ""
	return {
		"answer": f"{title}:\n\n{header}\n{sep}\n" + "\n".join(body) + more + passed_over,
		"doctype": doctype,
		"names": [r.name for r in rows],
	}


def _tidy_filters(doctype: str, filters: dict | None) -> dict | None:
	"""Filters as read() tidies them (labels, operator words), plus Select values matched to
	their option regardless of case ("overdue" -> "Overdue")."""
	if not filters:
		return filters
	args = {"doctype": doctype, "filters": dict(filters)}
	_builtins()._precheck_query(args)
	meta = frappe.get_meta(doctype)
	tidy = {}
	for key, value in (args["filters"] or {}).items():
		field = meta.get_field(key)
		value = _date_words(value)
		if field and field.fieldtype == "Select" and isinstance(value, str):
			options = {o.strip().lower(): o.strip() for o in (field.options or "").split("\n") if o.strip()}
			if value.strip().lower() not in options:
				# Report it rather than return "no records": the model can retry with a real value.
				raise ValueError(
					f"{value!r} is not a {field.label} of {doctype}. Options: {', '.join(options.values())}."
				)
			value = options[value.strip().lower()]
		tidy[key] = value
	return tidy


DATE_WORDS = {"today()": "today", "today": "today", "now()": "today", "now": "today", "nowdate()": "today"}


def _date_words(value: Any) -> Any:
	"""Date words a model writes into filters ("today()", ["<", "today"]) as the actual date."""
	if isinstance(value, str) and value.strip().lower() in DATE_WORDS:
		return frappe.utils.today()
	if isinstance(value, list) and len(value) == 2:
		return [value[0], _date_words(value[1])]
	return value


def _fmt(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, float):
		return f"{value:,.2f}"
	return str(value).replace("|", "/").replace("\n", " ")[:60]


# ── explain_doctype ───────────────────────────────────────────────────────────────────────


def _route_explain_doctype(text: str) -> dict[str, Any] | None:
	match = DEFINITION.match(text or "")
	doctype = match and _builtins()._doctype_from_phrase(match.group("what"), fuzzy=True)
	return {"doctype": doctype} if doctype else None


@tool(final_answer=True, route=_route_explain_doctype, named_in_conversation="doctype")
def explain_doctype(doctype: DocTypeArg) -> dict[str, Any]:
	"""Explain what one DocType is: module, purpose, lifecycle, main fields and links. Use for
	"what is <DocType>". Not for a module, app or feature as a whole ("how does payroll work"): use
	frappe_docs."""
	if not frappe.db.exists("DocType", doctype):
		# A concept ("valuation rate"), not a record type of this site: the docs define it.
		from flow.tools.web import docs_howto

		return docs_howto.func(topic=f"what is {doctype}")
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	meta = frappe.get_meta(doctype)
	route = doctype.lower().replace(" ", "-")
	parts = [f"**{doctype}** is a record type in the **{meta.module}** module."]
	if meta.description:
		parts.append(frappe.utils.strip_html(meta.description).strip())
	if meta.is_submittable:
		parts.append("It is **submittable**: saved as a Draft, then Submitted to make it final; a submitted record can only be cancelled (and amended), not edited.")
	if meta.is_tree:
		parts.append("Its records form a **tree** (groups containing records).")
	workflow = frappe.db.get_value("Workflow", {"document_type": doctype, "is_active": 1}, "workflow_name")
	if workflow:
		parts.append(f"It follows the **{workflow}** approval workflow.")

	main = _builtins()._user_filled(meta) + _builtins()._conditionally_required(meta)
	main_labels = [f.label for f in main if f.label][:MAIN_FIELDS_LIMIT]
	tables = [t.label for t in meta.get_table_fields() if t.reqd]
	if main_labels or tables:
		parts.append("Main fields: " + ", ".join(main_labels + [f"{t} (table)" for t in tables]) + ".")
	links = list(dict.fromkeys(f.options for f in meta.fields if f.fieldtype == "Link" and f.reqd and f.options))
	if links:
		parts.append("It links to: " + ", ".join(links) + ".")
	if meta.issingle:
		# A Single is one settings record with no table of its own: nothing to count.
		parts.append(f"It is a single settings page, not a list of records — [open it](/desk/{route}).")
	else:
		try:
			count = frappe.db.count(doctype)
		except Exception:
			count = None
		if count is not None:
			parts.append(f"There {'is' if count == 1 else 'are'} {count} on this site — [open the list](/desk/{route}).")
	return {"answer": "\n\n".join(parts), "doctype": doctype}


# ── low_stock ─────────────────────────────────────────────────────────────────────────────


def _below_reorder(levels: list[dict[str, Any]], quantities: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
	"""The reorder rows (item, warehouse, level) whose stock in that warehouse is at or below the
	level, lowest stock first. A warehouse with no stock record counts as zero."""
	low = []
	for row in levels:
		have = float(quantities.get((row["item"], row["warehouse"]), 0) or 0)
		if have <= float(row["level"] or 0):
			low.append({**row, "have": have})
	return sorted(low, key=lambda r: (r["have"] - float(r["level"] or 0), r["item"]))


@tool(final_answer=True)
def low_stock(
	limit: Annotated[int, "Most rows to show."] = LIST_LIMIT,
) -> dict[str, Any]:
	"""Items whose stock is at or below their reorder level, per warehouse. Use for "which items
	are low in stock", "what needs reordering", "running out of stock". Changes nothing."""
	for doctype in ("Item", "Bin"):
		if not frappe.db.exists("DocType", doctype):
			raise ValueError("Stock isn't set up on this site.")
		if not frappe.has_permission(doctype, "read"):
			raise PermissionError(f"No permission to read {doctype}")
	limit = min(max(int(limit or LIST_LIMIT), 1), LIST_ALL_LIMIT)
	levels = [
		{"item": r.parent, "warehouse": r.warehouse, "level": r.warehouse_reorder_level}
		for r in frappe.get_all(
			"Item Reorder",
			filters={"parenttype": "Item", "warehouse_reorder_level": [">", 0]},
			fields=["parent", "warehouse", "warehouse_reorder_level"],
			limit=MAX_STOCK_ROWS,
		)
	]
	if not levels:
		# Nothing says what "low" is here: show where stock is thinnest instead of guessing a level.
		thin = frappe.get_list(
			"Bin", filters={"actual_qty": [">", 0]}, fields=["item_code", "warehouse", "actual_qty"], order_by="actual_qty asc", limit=limit
		)
		if not thin:
			return {"answer": "There is no stock recorded on this site."}
		rows = "\n".join(
			f"| [{r.item_code}](/desk/item/{quote(str(r.item_code))}) | {r.warehouse} | {_fmt(r.actual_qty)} |" for r in thin
		)
		return {
			"answer": "No item has a reorder level set, so I can't say which are low. These have the least stock:\n\n"
			f"| Item | Warehouse | In stock |\n|---|---|---|\n{rows}"
		}
	bins = frappe.get_list(
		"Bin",
		filters={"item_code": ["in", list({r["item"] for r in levels})]},
		fields=["item_code", "warehouse", "actual_qty"],
		limit=MAX_STOCK_ROWS,
	)
	low = _below_reorder(levels, {(b.item_code, b.warehouse): b.actual_qty for b in bins})
	if not low:
		return {"answer": f"No item is at or below its reorder level ({len(levels)} reorder levels checked)."}
	rows = "\n".join(
		f"| [{r['item']}](/desk/item/{quote(str(r['item']))}) | {r['warehouse']} | {_fmt(r['have'])} | {_fmt(r['level'])} |"
		for r in low[:limit]
	)
	more = f"\n\nShowing {limit} of {len(low)}." if len(low) > limit else ""
	if len(levels) >= MAX_STOCK_ROWS:
		more += f"\n\nOnly the first {MAX_STOCK_ROWS} reorder levels were checked."
	return {
		"answer": f"**{len(low)} item{'s' if len(low) != 1 else ''} at or below the reorder level:**\n\n"
		f"| Item | Warehouse | In stock | Reorder level |\n|---|---|---|---|\n{rows}{more}"
	}


# ── on_leave ──────────────────────────────────────────────────────────────────────────────


@tool(final_answer=True)
def on_leave(
	date: Annotated[str | None, "Day to check, YYYY-MM-DD. Omit for today."] = None,
) -> dict[str, Any]:
	"""Who is on approved leave on a day (today by default), from the leave applications that cover
	it. Use for "who is on leave today", "who is off tomorrow". Changes nothing."""
	if not frappe.db.exists("DocType", "Leave Application"):
		raise ValueError("Leave management isn't set up on this site.")
	if not frappe.has_permission("Leave Application", "read"):
		raise PermissionError("No permission to read Leave Application")
	try:
		day = frappe.utils.getdate(date) if date else frappe.utils.getdate()
	except Exception:
		raise ValueError(f"{date!r} is not a date. Use YYYY-MM-DD.") from None
	rows = frappe.get_list(
		"Leave Application",
		filters={"status": "Approved", "docstatus": ["!=", 2], "from_date": ["<=", day], "to_date": [">=", day]},
		fields=["name", "employee", "employee_name", "leave_type", "from_date", "to_date"],
		order_by="employee_name asc",
		limit=LIST_ALL_LIMIT,
	)
	label = "today" if day == frappe.utils.getdate() else f"on {frappe.utils.formatdate(day)}"
	if not rows:
		return {"answer": f"No one is on approved leave {label}."}
	lines = "\n".join(
		f"| [{r.employee_name or r.employee}](/desk/leave-application/{quote(str(r.name))}) | {r.leave_type} | "
		f"{frappe.utils.formatdate(r.from_date)} to {frappe.utils.formatdate(r.to_date)} |"
		for r in rows
	)
	return {
		"answer": f"**{len(rows)} {'person is' if len(rows) == 1 else 'people are'} on approved leave {label}:**\n\n"
		f"| Employee | Leave type | Dates |\n|---|---|---|\n{lines}"
	}


# ── where_to_find ─────────────────────────────────────────────────────────────────────────


@tool(final_answer=True, named_in_conversation="doctype")
def where_to_find(doctype: DocTypeArg) -> dict[str, Any]:
	"""Where a record type is in the desk: its module, links to its list and to a new record, and
	the workspaces that link to it. Use when the user asks where to find, open or see a record
	type ("where can I find employees", "where do I see sales orders"). Not for the records
	themselves: use show_records."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	meta = frappe.get_meta(doctype)
	lines = [_("**{0}** is in the **{1}** module.").format(doctype, meta.module)]
	if meta.issingle:
		lines.append("- " + _("It is a single settings page: [open it]({0}).").format(_path(get_url_to_list(doctype))))
	else:
		lines.append("- " + _("See the records: [{0} list]({1}).").format(doctype, _path(get_url_to_list(doctype))))
		if frappe.has_permission(doctype, "create"):
			lines.append("- " + _("Add one: [New {0}]({1}).").format(doctype, _path(get_url_to_form(doctype, "new"))))
	for workspace, public in _workspaces_linking(doctype):
		lines.append("- " + _("It is also on the [{0}]({1}) workspace.").format(workspace, get_url_to_workspace(workspace, public)))
	return {"answer": "\n".join(lines), "doctype": doctype}


def _path(url: str) -> str:
	"""A link on this site as a path, so it works whatever address the user opened the site on."""
	return urlparse(url).path


def _workspaces_linking(doctype: str) -> list[tuple[str, bool]]:
	"""(name, is public) of the workspaces this user can open that have a link or shortcut to
	`doctype`."""
	found: list[str] = []
	for child in ("Workspace Link", "Workspace Shortcut"):
		if frappe.db.exists("DocType", child):
			found += frappe.get_all(child, filters={"link_to": doctype}, pluck="parent", limit=MAX_WORKSPACES)
	if not found:
		return []
	rows = frappe.get_list(
		"Workspace", filters={"name": ["in", list(dict.fromkeys(found))]}, fields=["name", "public"], limit=MAX_WORKSPACES
	)
	return [(row.name, bool(row.public)) for row in rows]


# ── document_flow ─────────────────────────────────────────────────────────────────────────


def _route_document_flow(text: str) -> dict[str, Any] | None:
	match = FLOW_QUESTION.match(text or "")
	doctype = match and _builtins()._doctype_from_phrase(match.group("what"), fuzzy=True)
	return {"doctype": doctype} if doctype else None


@tool(final_answer=True, route=_route_document_flow, named_in_conversation="doctype")
def document_flow(doctype: DocTypeArg) -> dict[str, Any]:
	"""Where a DocType sits in its business process: lifecycle, approval workflow, the documents
	it is made from and the ones made from it next. Use for "what happens after <DocType>"."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	meta = frappe.get_meta(doctype)
	parts = [f"**{doctype}** ({meta.module})"]
	if meta.is_submittable:
		parts.append("1. It is created as a **Draft** and can be edited.\n2. **Submit** makes it final: it can no longer be edited, only **Cancelled** (and then **Amended** into a new draft).")
	workflow = frappe.db.get_value("Workflow", {"document_type": doctype, "is_active": 1}, "name")
	if workflow:
		states = frappe.get_all("Workflow Document State", filters={"parent": workflow}, pluck="state", order_by="idx")
		parts.append(f"Approval workflow **{workflow}**: " + " → ".join(states) + ".")

	groups = _connections(meta)
	before = groups.get("before", [])
	after = groups.get("after", [])
	if before:
		parts.append("**Usually made from:** " + ", ".join(before) + ".")
	if after:
		parts.append("**What comes next** (documents created from it): " + ", ".join(after) + ".")
	if not (meta.is_submittable or workflow or before or after):
		parts.append("It is a standalone record: no lifecycle or connected documents are defined for it.")
	parts.append("Open a record and use the **Create** button to make the next document from it.")
	return {"answer": "\n\n".join(parts), "doctype": doctype}


def _connections(meta: Any) -> dict[str, list[str]]:
	"""Documents connected to this DocType, from its dashboard: "Reference" groups are where it
	came from, the rest (Related, Returns, ...) are made from it."""
	try:
		data = meta.get_dashboard_data()
	except Exception:
		return {}
	before, after = [], []
	for group in data.get("transactions") or []:
		items = [i for i in group.get("items") or [] if frappe.db.exists("DocType", i)]
		(before if (group.get("label") or "").lower() == "reference" else after).extend(items)
	return {"before": list(dict.fromkeys(before)), "after": list(dict.fromkeys(after))}


# ── assignment_steps ───────────────────────────────────────────────────────────────────────

# "how do I assign roles to employees", "how to link a contact with a customer",
# "steps to give a role to a user": acting on existing records rather than creating one.
ASSIGN_QUESTION = re.compile(
	r"^\s*(?:how\s+(?:do|can|should)\s+(?:i|we|you)|how\s+to|steps\s+(?:to|for)|what\s+are\s+the\s+steps\s+(?:to|for))\s+"
	r"(?:assign|link|attach|add|give|grant|allocate|map|connect|set)\s+(?P<item>[a-z][a-z \-]*?)\s+"
	r"(?:to|for|with|on|against)\s+(?P<target>[a-z][a-z \-]*?)\s*[?.!]*\s*$",
	re.IGNORECASE,
)
LINK_TYPES = ("Link", "Table", "Table MultiSelect")


def _route_assignment_steps(text: str) -> dict[str, Any] | None:
	match = ASSIGN_QUESTION.match(text or "")
	return match and assignment_arguments(match.group("item"), match.group("target"))


# Laya: an assign/link/attach verb, then "<item> to|with|for|on <target>" anywhere in the message
# ("can you tell me the way to attach a holiday list for employees").
LOOSE_ASSIGNMENT = re.compile(
	r"\b(?:assign|assigning|link|linking|attach|attaching|give|giving|grant|granting|allocate|allocating|map|mapping|connect|connecting|add|adding)\s+"
	r"(?P<item>[a-z][a-z \-]*?)\s+(?:to|for|with|on|against)\s+(?P<target>[a-z][a-z \-]*)",
	re.IGNORECASE,
)


def assignment_arguments(item_phrase: str, target_phrase: str, connected: bool = True) -> dict[str, Any] | None:
	"""{"item", "target"} when both phrases name DocTypes, and (with `connected`) the metadata can
	connect them, else None."""
	find = _builtins()._find_doctype_in_text
	item = find(item_phrase) or _assignment_doctype_for(item_phrase)
	target = find(target_phrase)
	if not item or not target or item == target or (connected and not _assignment_paths(item, target)):
		return None
	return {"item": item, "target": target}


def loose_assignment(text: str) -> dict[str, Any] | None:
	"""For a loosely worded question ("is it possible to assign an employee to a project?") that
	names two record types, the tool answers from this site's metadata even when nothing links them
	(it says so): a docs page about some other kind of assignment would be the wrong answer."""
	match = LOOSE_ASSIGNMENT.search(text or "")
	return match and assignment_arguments(match.group("item"), match.group("target"), connected=False)


@tool(
	final_answer=True,
	route=_route_assignment_steps,
	laya=LayaRoute("HOWTO", lambda text, signals: loose_assignment(text)),
)
def assignment_steps(
	item: Annotated[str, 'DocType being assigned or linked, e.g. "Role" in "assign roles to employees".'],
	target: Annotated[str, 'DocType it is assigned to, e.g. "Employee".'],
) -> dict[str, Any]:
	"""Steps to assign, link or attach one kind of record to another ("assign roles to
	employees"), found from field and table links in the metadata. Changes nothing."""
	for doctype in (item, target):
		if not frappe.has_permission(doctype, "read"):
			raise PermissionError(f"No permission to read {doctype}")
	paths = _assignment_paths(item, target)
	if not paths:
		return {
			"answer": f"This site has no field or table that links **{item}** to **{target}**, so they "
			"aren't assigned to each other directly. Ask your administrator if a custom field is needed.",
			"item": item,
			"target": target,
		}
	return {"answer": _describe_path(item, target, paths[0]), "item": item, "target": target}


ASSIGNMENT_SUFFIXES = ("Assignment", "Allocation")


def _assignment_doctype_for(phrase: str) -> str | None:
	"""An "<X> Assignment" / "<X> Allocation" DocType for a phrase that isn't a DocType itself
	("shift" -> Shift Assignment)."""
	words = " ".join(w.capitalize() for w in re.findall(r"[a-z]+", (phrase or "").lower()) if w not in ("a", "an", "the"))
	names = _builtins()._doctype_names()
	for suffix in ASSIGNMENT_SUFFIXES:
		for name in (f"{words} {suffix}", f"{words.rstrip('s')} {suffix}"):
			if name.lower() in names:
				return names[name.lower()]
	return None


def _assignment_paths(item: str, target: str) -> list[list[tuple[str, Any]]]:
	"""Ways to attach `item` to `target`, best first. Each path is a list of hops, every hop a
	(doctype the user opens, field they fill): a dedicated assignment record ("Salary Structure
	Assignment"), a field on the target, a field on the item pointing back at the target (a
	Dynamic Link table included, like Contact's Links), or a record the target links to."""
	if item.endswith(ASSIGNMENT_SUFFIXES) and _fields_to(item, target):
		return [[(item, f)] for f in _fields_to(item, target) if f.fieldtype == "Link"][:1] or []
	assignment = _assignment_record(item, target)
	if assignment:
		return [assignment]
	candidates = [(2, [(target, f)]) for f in _fields_to(target, item)]
	candidates += [(2, [(item, f)]) for f in _fields_to(item, target, dynamic=True) if f.fieldtype != "Link" or f.options == target]
	for link in frappe.get_meta(target).fields:
		if link.fieldtype != "Link" or not link.options or link.options in (item, target) or link.hidden:
			continue
		via = frappe.get_meta(link.options)
		if via.istable or via.issingle:
			continue
		candidates += [(0, [(target, link), (link.options, f)]) for f in _fields_to(link.options, item)]
	# Last resort: the item links to a record (an Employee's user) that a table of the target lists
	# (a Project's users), so the target takes the item through it.
	for link in frappe.get_meta(item).fields:
		if link.fieldtype != "Link" or not link.options or link.options in (item, target) or link.hidden:
			continue
		via = frappe.get_meta(link.options)
		if via.istable or via.issingle:
			continue
		candidates += [(-1, [(item, link), (target, f)]) for f in _fields_to(target, link.options) if f.fieldtype != "Link"]
	ranked = sorted(enumerate(candidates), key=lambda c: (-(c[1][0] + _fit(c[1][1][-1][1], item, target)), c[0]))
	return [path for _, (_, path) in ranked]


def _fit(field: Any, item: str, target: str) -> int:
	"""How plainly a field is "the place" for the assignment: named after the record type it
	holds ("roles" for Role, "department" for Department) rather than one special use of it
	("exception_budget_approver_role"), and a table, which can hold several."""
	held = field.options if field.fieldtype == "Link" else None
	names = {_snake(item), _snake(item) + "s", _snake(target), _snake(target) + "s"}
	score = 3 if field.fieldname in names or (held and field.fieldname == _snake(held)) else 0
	return score + (2 if field.fieldtype != "Link" else 0)


def _snake(doctype: str) -> str:
	return doctype.lower().replace(" ", "_")


def _assignment_record(item: str, target: str) -> list[tuple[str, Any]] | None:
	"""A "<item> Assignment" / "<item> Allocation" DocType that links to both: the way the apps
	mean the assignment to be made (it often carries dates and amounts the target can't)."""
	names = _builtins()._doctype_names()
	for suffix in ASSIGNMENT_SUFFIXES:
		doctype = names.get(f"{item} {suffix}".lower())
		if doctype and _fields_to(doctype, item) and _fields_to(doctype, target):
			return [(doctype, _fields_to(doctype, target)[0])]
	return None


def _fields_to(doctype: str, other: str, dynamic: bool = False) -> list[Any]:
	"""Fields of `doctype` that hold `other`: a visible Link to it, or a table whose rows link to
	it (with `dynamic`, also rows with a Dynamic Link, which can point at any DocType). Tables count
	even when hidden: some (User's roles) are edited through a custom widget."""
	found = []
	for f in frappe.get_meta(doctype).fields:
		if f.fieldtype not in LINK_TYPES or not f.options:
			continue
		if f.fieldtype == "Link":
			if f.options == other and not f.hidden:
				found.append(f)
			continue
		rows = frappe.get_meta(f.options).fields
		if any(c.fieldtype == "Link" and c.options == other for c in rows) or (
			dynamic and any(c.fieldtype == "Dynamic Link" for c in rows)
		):
			found.append(f)
	return found


def _describe_path(item: str, target: str, path: list[tuple[str, Any]]) -> str:
	steps = []
	if len(path) == 2 and path[0][0] == item:
		(_, link), (_, field) = path
		held = link.options
		steps.append(f"{item} has no field of its own on the {target}: they connect through the **{held}** the {item} links to.")
		steps.append(f"1. Open the {item} ([list](/desk/{_route(item)})) and check **{link.label}**: that's the {held} to add.")
		steps.append(f"2. Open the {target} ([list](/desk/{_route(target)})) and {_where(target, field)}{_fill(field, held)}.")
		steps.append(f"3. Save the {target}.")
	elif len(path) == 2:
		(_, link), (via, field) = path
		steps.append(
			f"{item} isn't set on the {target} itself: it's set on the **{via}** the {target} links to."
		)
		steps.append(
			f"1. Open the {target} ([list](/desk/{_route(target)})) and check **{link.label}**: that's its {via}. "
			f"If it's empty, pick or create the {via} there and save."
		)
		steps.append(f"2. Open that {via} and {_where(via, field)}{_fill(field, item)}.")
		steps.append(f"3. Save the {via}.")
	elif path[0][0] != target and path[0][0].endswith(ASSIGNMENT_SUFFIXES):
		((doctype, field),) = path
		if doctype != item:
			steps.append(f"Create a **{doctype}** record: it assigns the {item} to the {target}.")
		else:
			steps.append(f"Create a **{doctype}** record for the {target}.")
		steps.append(f"1. Open [New {doctype}](/desk/{_route(doctype)}/new).")
		chosen = f" and the {item}" if doctype != item else ""
		steps.append(f"2. Pick the {target} in **{field.label}**{chosen}, and fill in the other required fields.")
		steps.append(f"3. Save{' and Submit' if frappe.get_meta(doctype).is_submittable else ''} it.")
	else:
		((doctype, field),) = path
		other = target if doctype == item else item
		steps.append(f"1. Open the {doctype} ([list](/desk/{_route(doctype)})).")
		action = _where(doctype, field) + _fill(field, other)
		steps.append(f"2. {action[0].upper()}{action[1:]}.")
		steps.append(f"3. Save the {doctype}.")
	return "\n\n".join(steps)


def _fill(field: Any, other: str) -> str:
	if field.fieldtype == "Link":
		return f"set **{field.label}** to the {other}"
	if any(c.fieldtype == "Dynamic Link" for c in frappe.get_meta(field.options).fields):
		return f"add a row in the **{field.label}** table with {other} and the {other}'s name"
	return f"add the {other} in the **{field.label}** table"


def _where(doctype: str, field: Any) -> str:
	"""'in the <section> section, ' for the section holding `field`, when it has a label."""
	section = None
	for f in frappe.get_meta(doctype).fields:
		if f.fieldtype in ("Section Break", "Tab Break") and f.label:
			section = f.label
		if f.fieldname == field.fieldname:
			break
	return f"in the **{section}** section, " if section else ""


def _route(doctype: str) -> str:
	return doctype.lower().replace(" ", "-")


# ── status_steps ───────────────────────────────────────────────────────────────────────────

# "how can I make this invoice as paid", "how do I mark a sales order as completed",
# "how to set a task to closed", "how do I pay a purchase invoice"
STATUS_QUESTION = re.compile(
	r"^\s*(?:how\s+(?:do|can|should|would)\s+(?:i|we|you)|how\s+to|steps\s+(?:to|for))\s+"
	r"(?:(?:make|mark|set|get|turn|change|move|update)\s+(?:(?:this|that|the|a|an|my|our)\s+)?(?P<rest>[a-z][a-z0-9 ./\-]*?)"
	r"|(?P<verb>pay|close|cancel|submit|complete)\s+(?:(?:this|that|the|a|an|my|our)\s+)?(?P<what2>[a-z][a-z0-9 ./\-]*?))"
	r"\s*[?.!]*\s*$",
	re.IGNORECASE,
)
CONNECTORS = {"as", "to", "into", "be", "marked", "a", "an"}
# Verb -> the status it produces, and status word -> the stem of the documents that set it
# (English word forms, not site data).
VERB_STATUS = {"pay": "Paid", "close": "Closed", "cancel": "Cancelled", "submit": "Submitted", "complete": "Completed"}
STATUS_STEMS = {"paid": "pay", "partly paid": "pay", "billed": "bill", "delivered": "deliver", "received": "receiv",
	"returned": "return", "ordered": "order", "invoiced": "invoic", "transferred": "transfer"}


def _route_status_steps(text: str) -> dict[str, Any] | None:
	match = STATUS_QUESTION.match(text or "")
	if not match:
		return None
	if match.group("verb"):
		return status_arguments(match.group("what2"), VERB_STATUS.get(match.group("verb").lower()))
	return _split_status(match.group("rest"))


def _split_status(rest: str) -> dict[str, Any] | None:
	"""{"doctype", "status"} from "<record type> [as|to] <status>", trying the last one to three
	words as the status ("sales invoice as paid", "this sales order completed")."""
	words = (rest or "").split()
	for n in (1, 2, 3):
		if len(words) <= n:
			break
		head, status = words[:-n], " ".join(words[-n:])
		while head and head[-1].lower() in CONNECTORS:
			head = head[:-1]
		found = head and status_arguments(" ".join(head), status)
		if found:
			return found
	return None


def status_arguments(what_phrase: str, status_phrase: str) -> dict[str, Any] | None:
	"""{"doctype", "status"} when the phrase names a DocType that has that status, else None."""
	doctype = _builtins()._find_doctype_in_text(what_phrase or "")
	status = doctype and _status_option(doctype, status_phrase or "")
	return {"doctype": doctype, "status": status} if status else None


def _status_option(doctype: str, phrase: str) -> str | None:
	"""The DocType's own status option a phrase means ("paid" -> "Paid"), or lifecycle words."""
	field = frappe.get_meta(doctype).get_field("status")
	options = [o for o in ((field.options or "").split("\n") if field else []) if o]
	wanted = phrase.strip().lower()
	for option in options:
		if option.lower() == wanted:
			return option
	if frappe.get_meta(doctype).is_submittable and wanted in ("submitted", "cancelled", "canceled"):
		return "Cancelled" if wanted.startswith("cancel") else "Submitted"
	return None


@tool(
	final_answer=True,
	route=_route_status_steps,
	laya=LayaRoute("HOWTO", lambda text, signals: loose_status(text)),
)
def status_steps(
	doctype: DocTypeArg,
	status: Annotated[str, 'The status the user wants the record to reach, e.g. "Paid", "Completed".'],
) -> dict[str, Any]:
	"""How a record reaches a status ("mark this invoice as paid"): submit or cancel, or the
	connected document that sets it (Paid -> Payment Entry). Changes nothing."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	meta = frappe.get_meta(doctype)
	option = _status_option(doctype, status) or status
	route = _route(doctype)
	if option in ("Submitted", "Cancelled") and meta.is_submittable:
		verb = "Submit" if option == "Submitted" else "Cancel"
		steps = [f"1. Open the {doctype} ([list](/desk/{route})).", f"2. Click **{verb}** and confirm."]
		if verb == "Cancel":
			steps.append("Only a submitted record can be cancelled; to change it afterwards use **Amend**.")
		return {"answer": "\n\n".join(steps), "doctype": doctype, "status": option}

	creator = _status_creator(meta, option)
	status_field = meta.get_field("status")
	automatic = bool(status_field and status_field.read_only)
	if not creator:
		hint = "It is set by the system from linked documents" if automatic else f"Set **Status** to **{option}** and save"
		return {"answer": f"**{doctype}** has no connected document that sets **{option}**. {hint}.", "doctype": doctype, "status": option}

	steps = [f"A {doctype} becomes **{option}** when a **{creator}** is made against it."]
	n = 1
	if meta.is_submittable:
		steps.append(f"{n}. Open the {doctype} ([list](/desk/{route})); it must be **submitted** first (Submit if it is a Draft).")
	else:
		steps.append(f"{n}. Open the {doctype} ([list](/desk/{route})).")
	n += 1
	steps.append(f"{n}. Click **Create → {creator}**: it opens with the {doctype} already linked.")
	n += 1
	submit = " and **Submit**" if frappe.get_meta(creator).is_submittable else ""
	steps.append(f"{n}. Check the amounts and dates, then **Save**{submit}.")
	if automatic:
		steps.append(f"The {doctype}'s status then changes to **{option}** on its own (the Status field can't be edited by hand).")
	return {"answer": "\n\n".join(steps), "doctype": doctype, "status": option}


def _status_creator(meta: Any, option: str) -> str | None:
	"""The connected DocType whose record moves `meta` to `option`: from the dashboard group or
	item named like the status's verb ("Paid" -> the "Payment" group -> Payment Entry)."""
	stem = STATUS_STEMS.get(option.lower())
	if not stem:
		return None
	try:
		groups = meta.get_dashboard_data().get("transactions") or []
	except Exception:
		return None
	for group in groups:
		items = [i for i in group.get("items") or [] if frappe.db.exists("DocType", i)]
		if (group.get("label") or "").lower().startswith(stem) and items:
			return items[0]
	for group in groups:
		for item in group.get("items") or []:
			if stem in item.lower() and frappe.db.exists("DocType", item):
				return item
	return None


# Laya: "could you tell me the way to get this sales invoice marked as paid"
LOOSE_STATUS = re.compile(
	r"\b(?:make|mark|set|get|turn|change|move)\s+(?:(?:this|that|the|a|an|my|our)\s+)?(?P<rest>[a-z][a-z0-9 ./\-]*)"
	r"|\b(?P<verb>pay|close|cancel|submit)\s+(?:(?:this|that|the|a|an|my|our)\s+)?(?P<what2>[a-z][a-z0-9 ./\-]*)",
	re.IGNORECASE,
)


def loose_status(text: str) -> dict[str, Any] | None:
	match = LOOSE_STATUS.search(text or "")
	if not match:
		return None
	if match.group("verb"):
		return status_arguments(match.group("what2"), VERB_STATUS.get(match.group("verb").lower()))
	return _split_status(match.group("rest"))


# ── fields a question asks about ───────────────────────────────────────────────────────────

ASKED_FIELDS_LIMIT = 6


def asked_fields(doctype: str, text: str, exclude: str = "", record: str | None = None) -> list[str]:
	"""Fieldnames of `doctype` the question names by their label, best first.

	A field matches when a question word is one of its label's words (or shares a stem with one:
	"join" ~ "joining", "taxes" ~ "tax"). Fields are ranked by how much of their label the
	question covers ("which company" covers all of "Company", but little of "Use Company default
	Cost Center for Round off"), then by how many question words they match. With `record`, the
	best-ranked group with a value on that record is returned (an empty "Tax Id" gives way to
	"Total Taxes and Charges: 0.00"); words of the DocType's name and the record ID are ignored."""
	meta = frappe.get_meta(doctype)
	fields = [
		f for f in meta.fields
		if f.label and f.fieldtype not in _builtins().LAYOUT_FIELDTYPES and not f.permlevel and not f.hidden
	]
	if not fields:
		return []
	ignore = {_singular(w) for w in re.findall(r"[a-z]+", f"{doctype} {exclude}".lower())}
	label_words = {
		f.fieldname: {_singular(w) for w in re.findall(r"[a-z]+", f.label.lower())} - ignore for f in fields
	}
	asked = {
		_singular(w) for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and _singular(w) not in ignore
	}
	if not asked:
		return []

	def same(word: str, label_word: str) -> bool:
		return word == label_word or (
			min(len(word), len(label_word)) >= 4 and (label_word.startswith(word) or word.startswith(label_word))
		)

	ranked = []
	for f in fields:
		words = label_words[f.fieldname]
		if not words:
			continue
		covered = {lw for lw in words if any(same(w, lw) for w in asked)}
		if covered:
			ranked.append((len(covered) / len(words), len(covered), f.fieldname))
	if not ranked:
		return []
	ranked.sort(key=lambda r: (-r[0], -r[1]))
	tiers: list[list[str]] = []
	for coverage, matched, fieldname in ranked:
		if tiers and tiers[-1][0] == (coverage, matched):
			tiers[-1][1].append(fieldname)
		else:
			tiers.append([(coverage, matched), [fieldname]])
	if record:
		scalar_candidates = [
			n for _, names in tiers for n in names
			if meta.get_field(n) and meta.get_field(n).fieldtype not in ("Table", "Table MultiSelect")
		]
		scalar_vals: dict[str, Any] = {}
		if scalar_candidates:
			val = frappe.db.get_value(doctype, record, scalar_candidates, as_dict=True)
			if val:
				scalar_vals = dict(val)
		for _, names in tiers:
			found = False
			for n in names:
				f = meta.get_field(n)
				if not f:
					continue
				if f.fieldtype in ("Table", "Table MultiSelect"):
					if f.options and frappe.db.exists(f.options, {"parent": record, "parenttype": doctype, "parentfield": n}):
						found = True
						break
				else:
					v = scalar_vals.get(n)
					if f.fieldtype in ("Currency", "Float", "Int", "Percent"):
						if v is not None:
							found = True
							break
					elif f.fieldtype == "Check":
						if bool(v):
							found = True
							break
					elif v not in (None, ""):
						found = True
						break
			if found:
				return names[: ASKED_FIELDS_LIMIT * 2]
	return tiers[0][1][: ASKED_FIELDS_LIMIT * 2]


def _has_value(doc: Any, field: Any) -> bool:
	value = doc.get(field.fieldname)
	if field.fieldtype in ("Table", "Table MultiSelect"):
		return bool(value)
	if field.fieldtype in ("Currency", "Float", "Int", "Percent"):
		return value is not None
	if field.fieldtype == "Check":
		return bool(value)
	return value not in (None, "")


def _asked_values(doctype: str, name: str, fieldnames: list[str]) -> list[str]:
	"""Markdown lines for the asked fields that have a value on this record: a value per field,
	a small table for child tables (their list-view columns)."""
	if not fieldnames:
		return []
	meta = frappe.get_meta(doctype)
	table_fields = []
	scalar_fields = []
	for fn in fieldnames:
		field = meta.get_field(fn)
		if not field:
			continue
		if field.fieldtype in ("Table", "Table MultiSelect"):
			table_fields.append(fn)
		else:
			scalar_fields.append(fn)

	scalar_values: dict[str, Any] = {}
	if scalar_fields:
		val = frappe.db.get_value(doctype, name, scalar_fields, as_dict=True)
		if val:
			scalar_values = dict(val)

	lines = []
	for fieldname in fieldnames:
		field = meta.get_field(fieldname)
		if not field:
			continue
		if field.fieldtype in ("Table", "Table MultiSelect"):
			if not field.options:
				continue
			child = frappe.get_meta(field.options)
			cols = [c for c in child.fields if c.in_list_view and c.fieldtype not in _builtins().LAYOUT_FIELDTYPES][:4]
			if not cols:
				continue
			col_names = [c.fieldname for c in cols]
			rows = frappe.get_all(
				field.options,
				filters={"parent": name, "parenttype": doctype, "parentfield": fieldname},
				fields=col_names,
				order_by="idx asc",
			)
			if not rows:
				continue
			lines += [f"**{field.label}**:", "", "| " + " | ".join(c.label for c in cols) + " |", "|---" * len(cols) + "|"]
			lines += ["| " + " | ".join(_fmt(r.get(c.fieldname)) for c in cols) + " |" for r in rows]
			lines.append("")
		else:
			value = scalar_values.get(fieldname)
			if field.fieldtype in ("Currency", "Float", "Int", "Percent") and value is not None:
				# An amount of 0 answers the question ("no tax"), unlike an empty text field.
				lines.append(f"- **{field.label}**: {_fmt(value)}")
			elif value not in (None, "") and not (field.fieldtype == "Check" and not value):
				lines.append(f"- **{field.label}**: {_fmt(value)}")
		if len(lines) >= ASKED_FIELDS_LIMIT * 4:
			break
	return lines


# ── role_permissions ───────────────────────────────────────────────────────────────────────

ROLE_PERMISSIONS_LIMIT = 15


@tool(final_answer=True)
def role_permissions(
	role: Annotated[str, 'Role name, e.g. "Sales User".'],
	doctype: Annotated[str | None, "Limit to one DocType, e.g. \"Sales Invoice\". Omit for all."] = None,
) -> dict[str, Any]:
	"""What a role can do per DocType (read, write, create, submit…) and how to change it. Use
	for "give permissions to a role", "what can this role access", "let role X edit Y". Changes
	nothing."""
	if not frappe.has_permission("Role", "read"):
		raise PermissionError("No permission to read roles")
	if not frappe.db.exists("Role", role):
		# A DocType, a verb ("can create") or a pronoun ("me") here means the question is about
		# something else; an error lets the model pick the right tool instead of showing a
		# confident wrong answer.
		matches = frappe.get_all("Role", filters={"name": ["like", f"%{role}%"]}, pluck="name", limit=5) if re.search(r"[A-Za-z0-9]", role or "") else []
		hint = f" Close role names: {', '.join(matches)}." if matches else ""
		raise ValueError(
			f"{role!r} is not a role.{hint} For which roles can do something on a DocType use doctype_access; "
			"for the users holding a role use role_users; for a user's own roles use user_roles."
		)
	rights = [
		f for f in frappe.get_meta("DocPerm").fields
		if f.fieldtype == "Check" and f.fieldname not in ("if_owner",) and f.label
	]
	needed_fields = ["parent", *(f.fieldname for f in rights)]
	rows = _role_rules(role, doctype, fields=needed_fields)
	lines = []
	if rows:
		title = f"**{role}** can currently:" if not doctype else f"**{role}** on **{doctype}** can currently:"
		lines.append(title)
		for dt, rule in list(rows.items())[:ROLE_PERMISSIONS_LIMIT]:
			granted = [f.label for f in rights if rule.get(f.fieldname)]
			lines.append(f"- **{dt}**: {', '.join(granted) if granted else 'no rights ticked'}")
		if len(rows) > ROLE_PERMISSIONS_LIMIT:
			lines.append(f"…and {len(rows) - ROLE_PERMISSIONS_LIMIT} more DocTypes.")
	else:
		lines.append(f"**{role}** has no permissions{f' on **{doctype}**' if doctype else ''} yet.")

	page = frappe.db.get_value("Page", "permission-manager", ["name", "title"], as_dict=True)
	if page:
		target = f" and pick **{doctype}** as the Document Type" if doctype else ", pick the Document Type"
		lines += [
			"",
			f"To change them, open [{page.title}](/desk/{page.name}){target}:",
			f"1. If **{role}** has no row for it, click **Add A New Rule**, choose the role **{role}** and level 0.",
			f"2. Tick the rights it should have ({', '.join(f.label for f in rights[:6])}…).",
			"3. Changes are saved as you tick; users with the role get them on their next page load.",
		]
	return {"answer": "\n".join(lines), "role": role, "doctype": doctype}


# ── doctype_access ─────────────────────────────────────────────────────────────────────────

ACCESS_ROLES_LIMIT = 15
ACCESS_USERS_PER_ROLE = 10
# Who is asking "who"/"which user", and about access: the words that make a message a question
# about who may do something, as opposed to a request to list a record type.
WHO_ASKS = re.compile(r"\b(who|which\s+(?:users?|roles?|people|persons?)|what\s+(?:users?|roles?))\b", re.IGNORECASE)
ACCESS_WORDS = re.compile(r"\b(access|permissions?|allowed|rights?|authori[sz]ed|can|able|permit(?:ted)?)\b", re.IGNORECASE)


def _permission_rights() -> list[Any]:
	"""The rights a permission rule can grant (read, write, create…), from DocPerm's own fields."""
	return [f for f in frappe.get_meta("DocPerm").fields if f.fieldtype == "Check" and f.fieldname != "if_owner" and f.label]


def _named_right(text: str) -> str | None:
	"""The right a message names ("who can create…" -> "create"), by a DocPerm field's name or label."""
	words = set(re.findall(r"[a-z]+", (text or "").lower()))
	for field in _permission_rights():
		if field.fieldname in words or (field.label or "").lower() in words:
			return field.fieldname
	return None


def access_arguments(text: str) -> dict[str, Any] | None:
	"""doctype_access() arguments for "who can create a Sales Invoice" / "which users have access
	to Item", or None. The DocType must be named in the message."""
	if not (WHO_ASKS.search(text or "") and ACCESS_WORDS.search(text or "")):
		return None
	# "which users…" asks about people; without it, "users" would be read as the User DocType.
	doctype = _builtins()._find_doctype_in_text(WHO_ASKS.sub(" ", text))
	if not doctype:
		return None
	right = _named_right(text)
	return {"doctype": doctype, **({"right": right} if right else {})}


@tool(
	final_answer=True,
	route=lambda text: access_arguments(text),
	laya=LayaRoute("RECORD_LOOKUP", lambda text, signals: access_arguments(text)),
)
def doctype_access(
	doctype: DocTypeArg,
	right: Annotated[str | None, 'One right: "read", "write", "create", "delete", "submit", "cancel"… Omit for every right.'] = None,
) -> dict[str, Any]:
	"""Which roles, and which users through them, can create, read, write… a DocType ("who can
	create a Sales Invoice", "which users have access to Item"). Reads this site's permission
	rules and user roles. Changes nothing."""
	if not frappe.has_permission("Role", "read"):
		raise PermissionError("No permission to read roles")
	if not frappe.db.exists("DocType", doctype):
		raise ValueError(f"There is no DocType named {doctype}.")
	rights = _permission_rights()
	chosen = None
	if right:
		chosen = next((f for f in rights if right.strip().lower() in (f.fieldname, (f.label or "").lower())), None)
		if not chosen:
			raise ValueError(f"Unknown right {right!r}. Valid rights: {', '.join(f.fieldname for f in rights)}.")

	# A DocType's customised rules (Custom DocPerm) replace its standard ones, as Frappe applies them.
	fields = ["role", *(f.fieldname for f in rights)]
	rows = frappe.get_all("Custom DocPerm", filters={"parent": doctype, "permlevel": 0}, fields=fields)
	if not frappe.db.exists("Custom DocPerm", {"parent": doctype}):
		rows = frappe.get_all("DocPerm", filters={"parent": doctype, "permlevel": 0}, fields=fields)
	by_role: dict[str, list[str]] = {}
	for row in rows:
		granted = [f.label for f in rights if row.get(f.fieldname)]
		if (chosen and row.get(chosen.fieldname)) or (not chosen and granted):
			by_role.setdefault(row.role, [])
			by_role[row.role] += [g for g in granted if g not in by_role[row.role]]

	what = f"can **{chosen.label.lower()}**" if chosen else "have access to"
	if not by_role:
		return {"doctype": doctype, "right": right, "roles": [], "answer": f"No role {what} **{doctype}** in its permission rules. Only Administrator can."}

	users = _users_by_role(list(by_role)) if frappe.has_permission("User", "read") else None
	lines = [f"Roles that {what} **{doctype}**:", ""]
	for role in sorted(by_role)[:ACCESS_ROLES_LIMIT]:
		detail = f" ({', '.join(by_role[role])})" if not chosen else ""
		if users is None:
			lines.append(f"- **{role}**{detail}")
			continue
		names = users.get(role, [])
		shown = ", ".join(names[:ACCESS_USERS_PER_ROLE]) + (f" …and {len(names) - ACCESS_USERS_PER_ROLE} more" if len(names) > ACCESS_USERS_PER_ROLE else "")
		lines.append(f"- **{role}**{detail} — " + (f"{len(names)} user{'s' if len(names) != 1 else ''}: {shown}" if names else "no users"))
	if len(by_role) > ACCESS_ROLES_LIMIT:
		lines.append(f"…and {len(by_role) - ACCESS_ROLES_LIMIT} more roles.")
	lines += ["", "Administrator can always do everything."]
	if users is None:
		lines.append("(Your role can't list users, so only the roles are shown.)")
	return {"doctype": doctype, "right": right, "roles": sorted(by_role), "answer": "\n".join(lines)}


def _users_by_role(roles: list[str]) -> dict[str, list[str]]:
	"""{role: enabled system users who hold it}, without Administrator and Guest."""
	held = frappe.get_all("Has Role", filters={"parenttype": "User", "role": ["in", roles]}, fields=["parent", "role"], limit=MAX_ACCESS_ROWS)
	people = {
		u.name: u
		for u in frappe.get_all(
			"User",
			filters={"name": ["in", list({h.parent for h in held})], "enabled": 1, "user_type": "System User"},
			fields=["name", "full_name"],
			limit=MAX_ACCESS_ROWS,
		)
		if u.name not in ("Administrator", "Guest")
	}
	out: dict[str, list[str]] = {}
	for h in held:
		if h.parent in people:
			label = people[h.parent].full_name or h.parent
			out.setdefault(h.role, []).append(f"{label} ({h.parent})" if label != h.parent else h.parent)
	return {role: sorted(names) for role, names in out.items()}


# ── role_users / user_roles ────────────────────────────────────────────────────────────────

ROLE_USERS_LIMIT = 50
# "my roles", "which roles do I have": the asker's own roles.
OWN_ROLES = re.compile(r"\b(?:my\s+roles?|roles?\s+(?:do\s+)?(?:i|we)\s+(?:have|hold|got)|roles?\s+(?:am\s+i|i\s+am|i\s+was)\s+(?:given|assigned|in))\b", re.IGNORECASE)
# Asks about people ("who has…", "which users…"); "which roles…" asks about roles, not people.
ASKS_PEOPLE = re.compile(r"\b(who|(?:which|what)\s+(?:users?|people|persons?))\b", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _name_in_text(text: str, names: list[str]) -> str | None:
	"""The longest of `names` that appears in the text as whole words, ignoring case."""
	lowered = (text or "").lower()
	for name in sorted(filter(None, names), key=len, reverse=True):
		if re.search(rf"(?<![\w]){re.escape(name.lower())}(?![\w])", lowered):
			return name
	return None


def role_users_arguments(text: str) -> dict[str, Any] | None:
	"""role_users() arguments for "which users have the Sales Manager role", or None. The role
	must exist on this site and be named in the message; who-can-do-X-on-a-DocType questions
	belong to doctype_access."""
	if not ASKS_PEOPLE.search(text or "") or access_arguments(text):
		return None
	role = _name_in_text(WHO_ASKS.sub(" ", text), frappe.get_all("Role", pluck="name", limit=MAX_ACCESS_ROWS))
	return {"role": role} if role else None


def user_roles_arguments(text: str) -> dict[str, Any] | None:
	"""user_roles() arguments for "what roles do I have" (the person asking) or "what roles does
	<user> have", or None. Never an empty dict: routers read that as "no match"."""
	if not re.search(r"\broles?\b", text or "", re.IGNORECASE) or access_arguments(text):
		return None
	if OWN_ROLES.search(text):
		return {"user": None}
	email = EMAIL.search(text)
	if email:
		return {"user": email.group(0)}
	if not frappe.has_permission("User", "read"):
		return None
	users = frappe.get_all("User", filters={"enabled": 1, "user_type": "System User"}, fields=["name", "full_name"], limit=MAX_ACCESS_ROWS)
	label = _name_in_text(text, [u.full_name for u in users])
	return {"user": next(u.name for u in users if u.full_name == label)} if label else None


@tool(
	final_answer=True,
	route=lambda text: role_users_arguments(text),
	laya=LayaRoute("RECORD_LOOKUP", lambda text, signals: role_users_arguments(text)),
)
def role_users(role: Annotated[str, 'Role name, e.g. "Sales Manager".']) -> dict[str, Any]:
	"""The enabled users who hold a role ("which users have the Sales Manager role", "who is an
	Accounts User"). Not what the role can do (role_permissions). Changes nothing."""
	if not (frappe.has_permission("Role", "read") and frappe.has_permission("User", "read")):
		raise PermissionError("No permission to read roles and users")
	if not frappe.db.exists("Role", role):
		matches = frappe.get_all("Role", filters={"name": ["like", f"%{role}%"]}, pluck="name", limit=5)
		hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
		raise ValueError(f"There is no role named {role}.{hint}")
	names = _users_by_role([role]).get(role, [])
	if not names:
		return {"role": role, "users": [], "answer": f"No enabled user holds the role **{role}**."}
	more = f"\n…and {len(names) - ROLE_USERS_LIMIT} more." if len(names) > ROLE_USERS_LIMIT else ""
	lines = [f"**{len(names)}** user{'s' if len(names) != 1 else ''} hold the role **{role}**:", ""]
	lines += [f"- {name}" for name in names[:ROLE_USERS_LIMIT]]
	return {"role": role, "users": names, "answer": "\n".join(lines) + more}


@tool(
	final_answer=True,
	route=lambda text: user_roles_arguments(text),
	laya=LayaRoute("RECORD_LOOKUP", lambda text, signals: user_roles_arguments(text)),
)
def user_roles(
	user: Annotated[str | None, "User ID (email). Omit for the person asking."] = None,
) -> dict[str, Any]:
	"""The roles a user has ("what roles do I have", "which roles does <user> have"). Omit `user` for
	the person asking. Not what a role can do (role_permissions). Changes nothing."""
	from frappe.permissions import AUTOMATIC_ROLES

	own = not user or user == frappe.session.user
	if not own and not frappe.has_permission("User", "read"):
		raise PermissionError("No permission to read users")
	target = frappe.session.user if own else user
	if not frappe.db.exists("User", target):
		raise ValueError(f"There is no user {user}.")
	roles = sorted(set(frappe.get_roles(target)) - set(AUTOMATIC_ROLES))
	who = "You have" if own else f"**{target}** has"
	if not roles:
		return {"user": target, "roles": [], "answer": f"{who} no roles besides the automatic ones."}
	return {"user": target, "roles": roles, "answer": f"{who} **{len(roles)}** role{'s' if len(roles) != 1 else ''}:\n\n" + "\n".join(f"- {r}" for r in roles)}


def _role_rules(role: str, doctype: str | None, fields: list[str] | None = None) -> dict[str, dict[str, Any]]:
	"""{DocType: rule} for the role at permission level 0. A DocType's customised rules (Custom
	DocPerm) replace its standard ones (DocPerm), as Frappe applies them."""
	filters: dict[str, Any] = {"role": role, "permlevel": 0}
	custom_filters: dict[str, Any] = {}
	if doctype:
		filters["parent"] = doctype
		custom_filters["parent"] = doctype
	rules: dict[str, dict[str, Any]] = {}
	query_fields = fields or ["*"]
	customised = set(frappe.get_all("Custom DocPerm", filters=custom_filters or None, pluck="parent", distinct=True))
	for row in frappe.get_all("Custom DocPerm", filters=filters, fields=query_fields, order_by="parent"):
		rules[row.parent] = row
	for row in frappe.get_all("DocPerm", filters=filters, fields=query_fields, order_by="parent"):
		if row.parent not in customised:
			rules.setdefault(row.parent, row)
	return dict(sorted(rules.items()))

