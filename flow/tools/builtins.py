# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

from __future__ import annotations

import json
import re
from urllib.parse import quote
from typing import Annotated, Any, Literal

import frappe
from frappe import _

from flow.lib.tool import LayaRoute, LayaSignals, Tool, tool
from flow.utils.safe_exec import safe_exec

MAX_READ_LIMIT = 200
LAYOUT_FIELDTYPES = frozenset({"Section Break", "Column Break", "Tab Break", "HTML", "Heading"})
_CONFIRM_STR_LIMIT = 120
_ERROR_LIMIT = 300
_LIFECYCLE_BY_DOCSTATUS = {0: "submit", 1: "cancel", 2: "amend"}


def _summarize_values(values: dict) -> str:
	"""Truncate long values for confirm prompts — keeps the display scannable."""
	display = {}
	for k, v in (values or {}).items():
		if isinstance(v, str) and len(v) > _CONFIRM_STR_LIMIT:
			display[k] = v[:_CONFIRM_STR_LIMIT] + f"… ({len(v)} chars)"
		elif isinstance(v, list) and len(v) > 6:
			display[k] = [*v[:6], f"… +{len(v) - 6} more"]
		else:
			display[k] = v
	return json.dumps(display, indent=2, default=str, ensure_ascii=False)


# Parameter types shared by the tools below. The description travels with each parameter
# in the schema the model sees — small models lean on these far more than on the tool text.
DocTypeArg = Annotated[str, 'Exact DocType name, e.g. "Sales Invoice", "Customer", "Item".']
NamesArg = Annotated[
	list[str],
	"Record IDs exactly as count/read returned them — never make them up.",
]
FiltersArg = Annotated[
	dict[str, Any] | None,
	'Conditions as a JSON object, e.g. {"status": "Paid"}. Omit for all records.',
]


@tool
def find_doctypes(
	search: Annotated[str | None, 'Part of the name, e.g. "invoice".'] = None,
	module: Annotated[str | None, "Module; usually omit."] = None,
	limit: Annotated[int, "Max results."] = 40,
) -> list[dict]:
	"""Find a DocType's (record type's) exact name by keyword. Returns types, not records."""
	limit = min(max(int(limit), 1), MAX_READ_LIMIT)
	filters: dict[str, Any] = {"istable": 0}
	if module:
		filters["module"] = module
	if search:
		filters["name"] = ["like", f"%{search}%"]
	rows = frappe.get_all("DocType", filters=filters, fields=["name", "module"], order_by="name", limit=limit)
	if not rows and module and search:
		# Models often guess the module wrong (Sales Order under "Stock"); an empty result then
		# reads as "doesn't exist". The name match alone is the stronger signal.
		del filters["module"]
		rows = frappe.get_all("DocType", filters=filters, fields=["name", "module"], order_by="name", limit=limit)
	return [r for r in rows if frappe.has_permission(r["name"], "read")]


@tool
def describe(
	doctype: DocTypeArg,
	name: Annotated[
		str | None,
		"A record ID, only to list that record's actions (submit, cancel, ...). Never the DocType name.",
	] = None,
) -> dict[str, Any]:
	"""Fields of a DocType (fieldname, label, type, required, options), your permissions, whether it is
	submittable and its workflow, so you can pick field names before create/update. Link fields hold the
	name of a record of their `options` DocType; Table fields hold a list of row objects of it. Not for
	values (use read) or user guides (use creation_steps)."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")

	meta = frappe.get_meta(doctype)
	fields = [
		{
			"fieldname": f.fieldname,
			"label": f.label,
			"type": f.fieldtype,
			"options": f.options,
			"required": bool(f.reqd),
		}
		for f in meta.fields
		if f.fieldtype not in LAYOUT_FIELDTYPES
	]
	permissions = {p: bool(frappe.has_permission(doctype, p)) for p in ("read", "write", "create", "delete")}
	result: dict[str, Any] = {
		"doctype": doctype,
		"fields": fields,
		"permissions": permissions,
		# Shape the user-facing steps: submittable docs need Submit after Save; a workflow adds approvals.
		"submittable": bool(meta.is_submittable),
		"workflow": frappe.db.get_value(
			"Workflow", {"document_type": doctype, "is_active": 1}, "workflow_name"
		),
	}

	if name:
		# Dict form: `exists(dt, dt)` short-circuits as a Single-doctype check and returns truthy.
		if not meta.issingle and not frappe.db.exists(doctype, {"name": name}):
			# Spell out the fix: the usual mistake is passing the DocType (or its label) as `name`.
			raise ValueError(
				f"No {doctype} record named {name!r}. `name` must be a record ID; "
				f'find IDs with read(doctype="{doctype}", fields=["name"]).'
			)
		if not frappe.has_permission(doctype, "read", name):
			raise PermissionError(f"No permission to read {doctype} {name}")
		doc = frappe.get_doc(doctype, name)
		result["name"] = doc.name
		result["docstatus"] = int(doc.docstatus)
		result["actions"] = _doc_actions(doc, meta)
	return result


@tool(precheck=lambda args, context=None: _precheck_query(args))
def read(
	doctype: DocTypeArg,
	filters: FiltersArg = None,
	fields: Annotated[list[str] | None, 'Fieldnames, e.g. ["status", "grand_total"]. Omit for key columns.'] = None,
	limit: Annotated[int, "Max records."] = 20,
	order_by: Annotated[str | None, 'e.g. "creation desc".'] = None,
	start: Annotated[int, "Offset for pagination (number of records to skip). Default 0."] = 0,
) -> list[dict]:
	"""Field values of records, as data for you to reason over (e.g. before an update). Not for
	showing lists to the user: use show_records."""
	limit = min(max(int(limit), 1), MAX_READ_LIMIT)
	start = max(int(start or 0), 0)
	return frappe.get_list(
		doctype,
		filters=filters,
		fields=fields or _key_columns(doctype),
		limit=limit,
		start=start,
		order_by=order_by,
	)


# Record attributes every DocType has, valid in filters and fields.
STANDARD_COLUMNS = frozenset({"name", "owner", "creation", "modified", "modified_by", "docstatus", "idx"})
# Operator words models use, as Frappe filter operators.
OPERATOR_ALIASES = {
	"eq": "=", "equals": "=", "==": "=", "is": "=",
	"ne": "!=", "neq": "!=", "not equals": "!=", "<>": "!=",
	"gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
	"contains": "like", "includes": "like",
	"notin": "not in", "nin": "not in",
}
QUERY_OPTIONS = ("order_by", "fields", "limit", "start")
KEY_COLUMNS_LIMIT = 6


def _key_columns(doctype: str) -> list[str]:
	"""The columns worth showing when the model asks for records without naming fields: the
	ID, title, status and the DocType's list-view columns (a bare list of IDs answers nothing)."""
	cache = frappe.flags.flow_key_columns = getattr(frappe.flags, "flow_key_columns", None) or {}
	if doctype in cache:
		return cache[doctype]
	meta = frappe.get_meta(doctype)
	picked = ["name"]
	for fieldname in (meta.title_field, "status"):
		if fieldname and meta.has_field(fieldname):
			picked.append(fieldname)
	for f in meta.fields:
		if (
			f.in_list_view
			and not f.permlevel
			and f.fieldtype not in LAYOUT_FIELDTYPES
			and f.fieldtype not in ("Table", "Table MultiSelect", "Text Editor", "HTML", "Image", "Attach Image")
		):
			picked.append(f.fieldname)
	result = list(dict.fromkeys(picked))[: KEY_COLUMNS_LIMIT + 1]
	cache[doctype] = result
	return result


def _precheck_query(args: dict[str, Any]) -> str | None:
	"""Tidy read/count arguments the way small models garble them: options inside `filters`
	(`order_by` there reads to Frappe as a field it may not access), operator words ("eq",
	"contains"), labels for fieldnames; then name any filter key that isn't a field."""
	import difflib

	doctype, filters = args.get("doctype"), args.get("filters")
	if not doctype or not frappe.db.exists("DocType", doctype) or not isinstance(filters, dict):
		return None
	for option in QUERY_OPTIONS:
		if option in filters:
			value = filters.pop(option)
			if option in args and not args.get(option):
				args[option] = value
	if "offset" in filters:
		offset_val = filters.pop("offset")
		if "start" in args and not args.get("start"):
			args["start"] = offset_val
	meta = frappe.get_meta(doctype)
	by_label = {(f.label or "").lower(): f.fieldname for f in meta.fields if f.label and f.fieldtype not in LAYOUT_FIELDTYPES}
	tidy: dict[str, Any] = {}
	unknown = []
	for key, condition in filters.items():
		fieldname = key if (meta.has_field(key) or key in STANDARD_COLUMNS) else by_label.get(str(key).lower())
		fieldname = fieldname or _select_field_for_value(meta, condition)
		if not fieldname:
			names = [f.fieldname for f in meta.fields if f.fieldtype not in LAYOUT_FIELDTYPES]
			close = difflib.get_close_matches(str(key).lower(), names, n=3, cutoff=0.5)
			unknown.append(f"{key!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
			continue
		tidy[fieldname] = _normalize_condition(condition)
	args["filters"] = tidy
	if unknown:
		status = meta.get_field("status")
		hint = (
			f' To filter by status use {{"status": ...}} with one of: {", ".join(o for o in status.options.split(chr(10)) if o)}.'
			if status and status.fieldtype == "Select" and status.options
			else ""
		)
		return (
			f"Not fields of {doctype}: {', '.join(unknown)}. `filters` only holds field conditions "
			'like {"status": "Paid"}; put order_by, fields and limit beside it.' + hint
		)
	return None


def _select_field_for_value(meta: Any, value: Any) -> str | None:
	"""The one Select field that has `value` as an option ({"payment_status": "Unpaid"} on a
	Sales Invoice -> status): small models invent the key but usually get the value right."""
	if not isinstance(value, str) or not value.strip():
		return None
	wanted = value.strip().lower()
	matches = [
		f.fieldname
		for f in meta.fields
		if f.fieldtype == "Select" and wanted in {o.strip().lower() for o in (f.options or "").split("\n")}
	]
	return matches[0] if len(matches) == 1 else None


def _normalize_condition(condition: Any) -> Any:
	if not (isinstance(condition, list) and len(condition) == 2 and isinstance(condition[0], str)):
		return condition
	operator, value = condition[0].strip().lower(), condition[1]
	operator = OPERATOR_ALIASES.get(operator, operator)
	if operator == "like" and isinstance(value, str) and "%" not in value:
		value = f"%{value}%"
	return [operator, value]


# Required fields ERPNext/Frappe fill in themselves on a new form — listing them as steps
# would send the user hunting for fields they never touch.
AUTO_FILLED_FIELDS = frozenset(
	{
		"naming_series",
		"currency",
		"conversion_rate",
		"plc_conversion_rate",
		"price_list_currency",
		"selling_price_list",
		"buying_price_list",
		"item_name",
		"uom",
		"stock_uom",
		"conversion_factor",
		# Computed from the other amounts and the currencies.
		"source_exchange_rate",
		"target_exchange_rate",
		# Accounts and cost centre default from the Company / Item settings.
		"debit_to",
		"credit_to",
		"income_account",
		"expense_account",
		"cost_center",
	}
)
# Line-item fields users always fill, even where the row doesn't strictly require them
# (a Sales Invoice row allows free-text lines, so its Item Code is optional).
KEY_ROW_FIELDS = ("item_code", "qty", "rate")
# How many of a Link field's records the steps show as choices / examples.
LINK_EXAMPLES = 3


GREETING = re.compile(
	r"^\s*(h+i+|he+y+|hel+o+|hiya|yo|namaste|good\s+(morning|afternoon|evening)|greetings)\b[\s!.,]*(there|flow)?[\s!.]*$",
	re.IGNORECASE,
)
THANKS = re.compile(r"^\s*(thanks?|thank\s+you|thx|ty|ok(ay)?|cool|great|nice|got\s+it)\b[\s!.,]*(so\s+much|a\s+lot|flow)?[\s!.]*$", re.IGNORECASE)
def _route_small_talk(text: str) -> dict[str, Any] | None:
	if GREETING.match(text or ""):
		return {"kind": "greeting"}
	if THANKS.match(text or ""):
		return {"kind": "thanks"}
	return None


@tool(
	final_answer=True,
	route=_route_small_talk,
	laya=LayaRoute("SMALL_TALK", lambda text, signals: {"kind": "greeting"}),
)
def small_talk(kind: Annotated[str, '"greeting" or "thanks".'] = "greeting") -> dict[str, Any]:
	"""Reply to a greeting ("How can I help you today?") or thanks. No data is read."""
	if kind == "thanks":
		return {"answer": "You're welcome! Anything else I can help with?"}
	first = (frappe.utils.get_fullname(frappe.session.user) or "").split(" ")[0]
	hello = f"Hi {first}!" if first and first not in ("Administrator", "Guest") else "Hi!"
	result: dict[str, Any] = {"answer": f"{hello} How can I help you today?"}
	# The browser model is several GB on first use: a greeting is the cue for the panel to
	# start fetching it, so the first real question doesn't wait on the download.
	from flow.assistant.default import DEFAULT_MODEL_ID
	from flow.lib.webllm import preload_hint

	preload = preload_hint(DEFAULT_MODEL_ID)
	if preload:
		result["preload"] = preload
	return result


@tool(
	final_answer=True,
	route=lambda text: _route_required_values(text),
	laya=LayaRoute("CREATE", lambda text, signals: _laya_required_values(text)),
)
def required_values(doctype: DocTypeArg) -> dict[str, Any]:
	"""Ask for the values needed to create a record of a DocType, with an example line to fill
	in. Use when the user wants to create one but gave no values."""
	if not frappe.has_permission(doctype, "create"):
		raise PermissionError(f"No permission to create {doctype}")
	meta = frappe.get_meta(doctype)
	date_format = frappe.db.get_single_value("System Settings", "date_format") or "yyyy-mm-dd"
	_defaults, sole_links = _field_facts(doctype)
	fields = [f for f in _user_filled(meta) if f.fieldname not in sole_links]
	fields += _conditionally_required(meta)
	rows = []
	for table in meta.get_table_fields():
		if table.reqd:
			child = frappe.get_meta(table.options)
			cols = [child.get_field(f) for f in KEY_ROW_FIELDS if child.get_field(f)]
			rows.append((table, cols))
	if not fields and not rows:
		fields = _important_fields(meta, exclude=set())
	lines = [f"- {_field_hint(f, date_format)}" for f in fields]
	for table, cols in rows:
		lines.append(f"- **{table.label}**: " + ", ".join(f.label for f in cols))
	example = ", ".join(
		[f"{(f.label or f.fieldname).lower()} …" for f in fields] + [f"{c.label.lower()} …" for _t, cols in rows for c in cols]
	)
	auto = ", ".join(
		f"**{meta.get_field(k).label}** ({v})" for k, v in sole_links.items() if meta.get_field(k)
	)
	answer = (
		f"To create a new **{doctype}**, I need:\n" + "\n".join(lines)
		+ (f"\n\n{auto} will be filled in automatically." if auto else "")
		+ f"\n\nSend them in one message, e.g.:\n> create {doctype.lower()} with {example}"
	)
	return {"doctype": doctype, "answer": answer}


@tool(
	final_answer=True,
	route=lambda text: _route_creation_steps(text),
	laya=LayaRoute("HOWTO", lambda text, signals: _laya_creation_steps(text)),
)
def creation_steps(doctype: DocTypeArg) -> dict[str, Any]:
	"""Steps to create a NEW record of a DocType by hand in the desk, from this site's required
	fields. Use for "how do I create/add a <DocType>". Not for assigning, linking, status changes or
	other how-tos, and not to re-explain an earlier answer. Creates nothing."""
	if not frappe.has_permission(doctype, "read"):
		raise PermissionError(f"No permission to read {doctype}")
	meta = frappe.get_meta(doctype)
	route = doctype.lower().replace(" ", "-")
	date_format = frappe.db.get_single_value("System Settings", "date_format") or "yyyy-mm-dd"

	steps = [
		f"Open [New {doctype}](/desk/{route}/new), or search **{doctype}** in the search bar "
		f"and click **+ Add {doctype}**."
	]
	fields = _user_filled(meta)
	conditional = _conditionally_required(meta)
	important = _important_fields(meta, exclude={f.fieldname for f in fields + conditional})
	if fields:
		steps.append("Fill in the required fields:\n" + _bullets(fields, date_format))
	if conditional:
		steps.append("Required in some cases (the form marks them when they apply):\n" + _bullets(conditional, date_format))
	if important:
		steps.append("Usually also needed:\n" + _bullets(important, date_format))
	for table in meta.get_table_fields():
		if not table.reqd:
			continue
		child_meta = frappe.get_meta(table.options)
		key = [child_meta.get_field(f) for f in KEY_ROW_FIELDS if child_meta.get_field(f)]
		cols = key + [f for f in _user_filled(child_meta) if f.fieldname not in KEY_ROW_FIELDS]
		cols += _important_fields(child_meta, exclude={f.fieldname for f in cols}, limit=2)
		steps.append(
			f"In the **{table.label}** table, add a row for each entry and fill:\n" + _bullets(cols, date_format)
		)
	steps.append("Click **Save** (Ctrl+S).")
	if meta.is_submittable:
		steps.append("Click **Submit** to finalise it. A submitted record can only be cancelled, not edited.")
	workflow = frappe.db.get_value("Workflow", {"document_type": doctype, "is_active": 1}, "workflow_name")
	if workflow:
		steps.append(f"Move it through the **{workflow}** workflow using the **Actions** button.")

	numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
	answer = (
		f"**How to create a new {doctype}**\n\n{numbered}\n\n"
		"All other fields are optional."
	)
	return {"doctype": doctype, "answer": answer}


def _label_of(doctype: str, fieldname: str) -> str:
	field = frappe.get_meta(doctype).get_field(fieldname)
	return (field.label if field else None) or fieldname


def _bullets(fields: list[Any], date_format: str) -> str:
	return "\n".join(f"   - {_field_hint(f, date_format)}" for f in fields)


def _editable(f: Any) -> bool:
	return (
		f.fieldtype not in LAYOUT_FIELDTYPES
		and f.fieldtype not in ("Table", "Table MultiSelect", "Check", "Button", "HTML", "Image")
		and not (f.hidden or f.read_only or f.fetch_from)
		and f.fieldname not in AUTO_FILLED_FIELDS
	)


def _shown_on_new(meta: Any, field: Any) -> bool:
	"""Whether a field is visible on a fresh form, by evaluating its `depends_on` against a new
	record with its defaults (Sales Order's Delivery Date: `!doc.skip_delivery_note` -> shown;
	a Payment Entry's cheque number: `doc.paid_from && doc.paid_to` -> hidden). The JS-style
	expression is translated for safe_eval; anything that won't evaluate counts as shown."""
	return _holds_on_new(meta, field.depends_on)


def _holds_on_new(meta: Any, condition: str | None) -> bool:
	"""Evaluate a form condition (`depends_on` style: a fieldname or `eval:<JS>`) against a new
	record's defaults. Empty or unevaluable conditions count as true."""
	condition = (condition or "").strip()
	if not condition:
		return True
	doc = _new_doc_values(meta.name)
	if not condition.startswith("eval:"):
		return bool(doc.get(condition))
	expr = condition[len("eval:") :].replace("?.", ".")
	expr = re.sub(r"\.length\b", "", expr)  # list/str truthiness stands in for .length
	for js, py in (("===", "=="), ("!==", "!="), ("&&", " and "), ("||", " or ")):
		expr = expr.replace(js, py)
	expr = re.sub(r"!(?!=)", " not ", expr)
	expr = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", expr))
	expr = re.sub(r"\b(null|undefined)\b", "None", expr)
	try:
		return bool(
			frappe.safe_eval(
				expr.strip(), None, {"doc": doc, "in_list": lambda options, value: value in options}
			)
		)
	except Exception:
		return True


def _new_doc_values(doctype: str) -> Any:
	"""A new record's default values, cached for the request."""
	cache = frappe.flags.flow_new_doc_values = frappe.flags.flow_new_doc_values or {}
	if doctype not in cache:
		try:
			cache[doctype] = frappe._dict(frappe.new_doc(doctype).as_dict())
		except Exception:
			cache[doctype] = frappe._dict()
	return cache[doctype]


def _conditionally_required(meta: Any) -> list[Any]:
	"""Fields made mandatory by a condition (`mandatory_depends_on`), e.g. an Address Title
	when no link is set, or a Lead's First Name when there is no Company Name."""
	# Skipped when the field is itself hidden behind a display condition (POS-only times,
	# cheque details for bank payments): those surface on the form once they apply.
	return [
		f
		for f in meta.fields
		if f.mandatory_depends_on
		and not f.reqd
		and _editable(f)
		and _shown_on_new(meta, f)
		# Required on a typical new record: a Lead's First Name (no Company Name yet) is,
		# an Item's Shelf Life (only for batch items with expiry) is not.
		and _holds_on_new(meta, f.mandatory_depends_on)
	]


def _important_fields(meta: Any, exclude: set[str], limit: int = 4) -> list[Any]:
	"""Optional fields the DocType flags as prominent (shown in its list view, or bold) — the
	usual home of fields enforced in code rather than marked required, like a Sales Order's
	Delivery Date or a Payment Entry's Party."""
	picked = [
		f
		for f in meta.fields
		if (f.in_list_view or f.bold)
		and not f.reqd
		and not f.default
		and f.fieldname not in exclude
		and f.fieldtype not in ("Currency", "Float", "Int", "Percent")  # totals, usually computed
		and (f.fieldtype == "Dynamic Link" or _shown_on_new(meta, f))  # hidden on a fresh form
		and _editable(f)
	][:limit]
	# A Dynamic Link is meaningless without the field naming its DocType (Party -> Party Type).
	for f in list(picked):
		if f.fieldtype == "Dynamic Link" and f.options not in exclude:
			companion = meta.get_field(f.options)
			if companion and companion not in picked:
				picked.insert(picked.index(f), companion)
	return picked


def _user_filled(meta: Any) -> list[Any]:
	"""Required fields the user has to fill on a new record (no default, not auto-filled)."""
	return [
		f
		for f in meta.fields
		if f.reqd
		and f.fieldtype not in LAYOUT_FIELDTYPES
		and f.fieldtype not in ("Table", "Table MultiSelect")
		and not (f.hidden or f.read_only or f.default or f.fetch_from)
		and f.fieldname not in AUTO_FILLED_FIELDS
	]


def _field_hint(field: Any, date_format: str = "yyyy-mm-dd") -> str:
	"""A field's bold label plus, where useful, the values it accepts."""
	label = f"**{field.label or field.fieldname}**"
	if field.fieldtype == "Select" and field.options:
		choices = [o for o in field.options.split("\n") if o.strip()]
		return f"{label}: one of {', '.join(choices)}"
	if field.fieldtype == "Dynamic Link" and field.options:
		return f"{label}: a record of the type chosen in **{_label_of(field.parent, field.options)}**"
	if field.fieldtype == "Link" and field.options == "DocType":
		return f"{label}: a document type, e.g. Customer"
	if field.fieldtype == "Link" and field.options:
		cache = frappe.flags.flow_link_examples = getattr(frappe.flags, "flow_link_examples", None) or {}
		if field.options in cache:
			names = cache[field.options]
		else:
			target = frappe.get_meta(field.options)
			leaf_only = {"is_group": 0} if target.is_tree and target.has_field("is_group") else None
			names = frappe.get_all(
				field.options, filters=leaf_only, pluck="name", order_by="creation asc", limit=LINK_EXAMPLES + 1
			)
			cache[field.options] = names
		if len(names) == 1:
			return f"{label}: {names[0]}"
		if names and len(names) <= LINK_EXAMPLES:
			return f"{label}: one of {', '.join(names)}"
		if names:
			return f"{label}: pick from the list, e.g. {', '.join(names[:LINK_EXAMPLES])}"
		return f"{label}: pick an existing {field.options}"
	if field.fieldtype in ("Date", "Datetime"):
		return f"{label}: a date ({date_format})"
	if field.fieldtype == "Data" and field.options in ("Email", "Phone", "URL"):
		return f"{label}: {'a phone number' if field.options == 'Phone' else 'an ' + field.options.lower() if field.options == 'Email' else 'a URL'}"
	return label


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


def _laya_creation_steps(text: str) -> dict[str, Any] | None:
	# HOWTO covers any "how do I ..." but creation_steps only answers how to create a record,
	# so it takes the message only when it asks to create the DocType it names.
	doctype = _creation_target(text)
	return {"doctype": doctype} if doctype else None


def _laya_required_values(text: str) -> dict[str, Any] | None:
	# Laya never routes to create itself: only the read-only "what values do you need" reply for
	# a request that names a DocType but gives no values.
	parsed = _parse_create_request(text)
	return {"doctype": parsed[0]} if parsed and not parsed[1] else None


def _laya_count(text: str, signals: LayaSignals) -> dict[str, Any] | None:
	# The same reading as the exact route; a breakdown or scope it can't map to a field goes to
	# the chat model rather than becoming a plain total.
	from flow.tools.answers import count_arguments

	return count_arguments(text) if signals.asks_count >= signals.threshold else None


COUNT_NAMES_LIMIT = 20


@tool(
	final_answer=True,
	precheck=lambda args, context=None: _precheck_query(args),
	route=lambda text: _route_count(text),
	laya=LayaRoute("RECORD_LOOKUP", lambda text, signals: _laya_count(text, signals)),
)
def count(
	doctype: DocTypeArg,
	filters: FiltersArg = None,
	group_by: Annotated[str | None, 'Fieldname to count per value ("stage wise", "by status"), e.g. "status". Omit for one total.'] = None,
) -> dict[str, Any]:
	"""Count records ("how many"), optionally per value of one field ("leads by status")."""
	if frappe.get_meta(doctype).issingle:
		raise ValueError(f"{doctype} is a single settings page, not a list of records; there is nothing to count.")
	if group_by:
		return _grouped_count(doctype, filters, group_by)
	rows = frappe.get_list(doctype, filters=filters, fields=[{"COUNT": "*", "as": "count"}])
	total = int(rows[0]["count"]) if rows else 0
	# Real IDs up front: small models otherwise invent plausible-looking record names.
	names = (
		frappe.get_list(
			doctype, filters=filters, pluck="name", order_by="creation desc", limit=COUNT_NAMES_LIMIT
		)
		if total > 0
		else []
	)
	return {"doctype": doctype, "count": total, "names": names, "answer": _count_answer(doctype, total, names)}


# "how many employees (are there)?" / "how many sales invoices in the system"
COUNT_QUESTION = re.compile(
	r"^\s*(?:how\s+many|count(?:\s+of)?(?:\s+the)?|number\s+of)\s+(?:total\s+)?(?P<what>[a-z][a-z \-]*?)"
	r"(?:\s*[?.!,]*\s+(?:are|is|do|does|exist|exists|present|there|in|we|i|have)\b.*)?\s*[?.!]*\s*$",
	re.IGNORECASE,
)
# "how to create a new employee?", "how do i add a sales order give me the steps",
# "steps to create an item" — but not "create a todo with description X" (that's a create).
HOW_TO_QUESTION = re.compile(
	r"^\s*(?:how\s+(?:do|can|should)\s+(?:i|we|you)|how\s+to|steps\s+(?:to|for)|give\s+me\s+(?:the\s+)?steps\s+(?:to|for))\s+"
	r"(?:create|add|make|enter|register|set\s+up|new)\s+(?:(?:a|an|the)\s+)?(?:new\s+)?(?P<what>[a-z][a-z \-]*?)"
	r"(?:\s*[?.!,]*\s+(?:in|on|record|entry|give|with|step|steps|please)\b.*)?\s*[?.!]*\s*$",
	re.IGNORECASE,
)


# "create a todo with description X", "add new customer named Y", "create one employee"
CREATE_REQUEST = re.compile(
	r"^\s*(?:please\s+)?(?:create|add|make|register|enter)\s+(?:(?:a|an|one|1|the|new)\s+)*(?P<rest>.+?)\s*[.!]*\s*$",
	re.IGNORECASE | re.DOTALL,
)
VALUE_SEPARATORS = re.compile(r"\s*(?:[,;\n]|\band\b)\s*", re.IGNORECASE)


def _route_create(text: str) -> dict[str, Any] | None:
	"""A create request whose values all parse as "<field label> <value>" pairs of the DocType
	(item-table fields included): the record, ready for create's checks and the approval card."""
	parsed = _parse_create_request(text)
	if not parsed or not parsed[1]:
		return None
	doctype, record = parsed
	return {"doctype": doctype, "records": [record]}


def _route_required_values(text: str) -> dict[str, Any] | None:
	"""A create request naming only the DocType ("create one employee"): ask for the values."""
	parsed = _parse_create_request(text)
	if not parsed or parsed[1]:
		return None
	return {"doctype": parsed[0]}


def _parse_create_request(text: str) -> tuple[str, dict[str, Any]] | None:
	"""(DocType, record) for "create <doctype> [with|for] <label> <value>, ...", record empty when
	no values are given; None when the DocType or any value part isn't recognised."""
	match = CREATE_REQUEST.match(text or "")
	if not match:
		return None
	words = match.group("rest").split()
	for n in range(min(4, len(words)), 0, -1):
		doctype = _doctype_from_phrase(" ".join(words[:n]).strip(":,;-"))
		if doctype:
			break
	else:
		return None
	rest = " ".join(words[n:]).strip()
	rest = re.sub(r"^(?:with|for|having|where|:|-)\s+", "", rest, flags=re.IGNORECASE)
	if not rest or rest.lower() in ("please", "for me", "now"):
		return doctype, {}
	record = _parse_values(doctype, rest)
	return (doctype, record) if record else None


def _parse_values(doctype: str, text: str) -> dict[str, Any] | None:
	"""Split "customer test, item X, quantity 3" into fields by matching each part's start to a
	field label; item-table fields go into one row of their table. None if any part is unknown."""
	meta = frappe.get_meta(doctype)
	targets: list[tuple[str, str, str | None]] = []  # (label, fieldname, table fieldname)
	for f in meta.fields:
		if f.label and _editable(f):
			targets.append((f.label.lower(), f.fieldname, None))
	for table in meta.get_table_fields():
		for f in frappe.get_meta(table.options).fields:
			if f.label and _editable(f):
				targets.append((f.label.lower(), f.fieldname, table.fieldname))
	targets.sort(key=lambda t: len(t[0]), reverse=True)  # longest label wins ("item code" over "item")

	record: dict[str, Any] = {}
	for part in filter(None, (p.strip() for p in VALUE_SEPARATORS.split(text))):
		lowered = part.lower()
		for label, fieldname, table in targets:
			if lowered.startswith(label) and (len(lowered) == len(label) or not lowered[len(label)].isalnum()):
				value = re.sub(r"^\s*(?:is|=|:|as|named|of|to)?\s*", "", part[len(label) :]).strip(" :=\"'")
				if not value:
					return None
				if table:
					rows = record.setdefault(table, [{}])
					rows[0].setdefault(fieldname, value)
				else:
					record.setdefault(fieldname, value)
				break
		else:
			return None
	return record


def _route_count(text: str) -> dict[str, Any] | None:
	# "how many leads in Frappe CRM", "what is the total leads? stage wise", "how many unpaid
	# sales invoices": parsed in answers.count_arguments, shared with Laya.
	from flow.tools.answers import count_arguments

	return count_arguments(text)


def _route_creation_steps(text: str) -> dict[str, Any] | None:
	match = HOW_TO_QUESTION.match(text)
	doctype = match and _doctype_from_phrase(match.group("what"), fuzzy=True)
	return {"doctype": doctype} if doctype else None


def _doctype_from_phrase(phrase: str, fuzzy: bool = False) -> str | None:
	"""The readable, non-child DocType a phrase names, allowing plurals ("sales invoices",
	"employees", "entries") and a trailing "record(s)". With `fuzzy`, also a misspelling
	("employyes", "sales invocies") when one DocType is clearly the closest; None otherwise."""
	words = re.sub(r"\s+", " ", (phrase or "").strip().lower())
	words = re.sub(r"\s+(records?|entries|documents?)$", "", words)
	if not words:
		return None
	candidates = [words]
	if words.endswith("ies"):
		candidates.append(words[:-3] + "y")
	if words.endswith("es"):
		candidates.append(words[:-2])
	if words.endswith("s"):
		candidates.append(words[:-1])
	names = _doctype_names()
	for candidate in candidates:
		doctype = names.get(candidate)
		if doctype and frappe.has_permission(doctype, "read"):
			return doctype
	if fuzzy:
		doctype = _closest_doctype(candidates, names)
		if doctype and frappe.has_permission(doctype, "read"):
			return doctype
	return None


# A misspelling must be this similar to a DocType name, and this much closer than the runner-up,
# to be read as that DocType ("employyes" -> Employee, but not "system" -> System Console).
FUZZY_MIN_RATIO = 0.85
FUZZY_MIN_LEAD = 0.05


def _closest_doctype(candidates: list[str], names: dict[str, str]) -> str | None:
	import difflib

	scores: dict[str, float] = {}
	for candidate in candidates:
		if len(candidate) < 5:  # short words are too easy to confuse
			continue
		for key in difflib.get_close_matches(candidate, names, n=5, cutoff=FUZZY_MIN_RATIO):
			ratio = difflib.SequenceMatcher(None, candidate, key).ratio()
			scores[key] = max(scores.get(key, 0.0), ratio)
	ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
	if not ranked or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < FUZZY_MIN_LEAD):
		return None
	return names[ranked[0][0]]


def _find_doctype_in_text(text: str) -> str | None:
	"""The DocType a free-form message names ("how do I raise a sales invoice for test?" ->
	Sales Invoice): the longest run of up to 4 consecutive words that is a DocType name."""
	words = re.findall(r"[a-z][a-z\-]*", (text or "").lower())
	for size in range(min(4, len(words)), 0, -1):
		for start in range(len(words) - size + 1):
			doctype = _doctype_from_phrase(" ".join(words[start : start + size]))
			if doctype:
				return doctype
	return None


def _doctype_names() -> dict[str, str]:
	"""lower-cased name -> name of every non-child DocType, cached for the request."""
	cache = frappe.flags.flow_doctype_names
	if cache is None:
		cache = frappe.flags.flow_doctype_names = {
			n.lower(): n for n in frappe.get_all("DocType", filters={"istable": 0}, pluck="name")
		}
	return cache


def _grouped_count(doctype: str, filters: dict | None, group_by: str) -> dict[str, Any]:
	meta = frappe.get_meta(doctype)
	field = meta.get_field(group_by)
	if not field and group_by not in STANDARD_COLUMNS:
		raise ValueError(f"{group_by!r} is not a field of {doctype}.")
	label = field.label if field else group_by
	rows = frappe.get_list(
		doctype, filters=filters, fields=[group_by, {"COUNT": "*", "as": "count"}], group_by=group_by,
		order_by="count desc",
	)
	total = sum(int(r["count"]) for r in rows)
	if not total:
		return {"doctype": doctype, "count": 0, "groups": {}, "answer": f"There are no **{doctype}** records."}
	groups = {(r.get(group_by) or "(not set)"): int(r["count"]) for r in rows}
	table = "\n".join(f"| {value} | {n} |" for value, n in groups.items())
	answer = f"**{total} {doctype}** record{'s' if total != 1 else ''} by **{label}**:\n\n| {label} | Count |\n|---|---|\n{table}"
	return {"doctype": doctype, "count": total, "groups": groups, "answer": answer}


def _count_answer(doctype: str, total: int, names: list[str]) -> str:
	route = doctype.lower().replace(" ", "-")
	if not total:
		return f"There are no **{doctype}** records."
	noun = doctype if total == 1 else f"{doctype} records"
	lines = []
	titles = _batch_display_names(doctype, names)
	for name in names:
		title = titles.get(name, name)
		label = f"{name} — {title}" if title != name else name
		# IDs can contain spaces ("Grace Park"); a raw space breaks the Markdown link.
		lines.append(f"- [{label}](/desk/{route}/{quote(str(name))})")
	more = f"\n…and {total - len(names)} more." if total > len(names) else ""
	return (
		f"There {'is' if total == 1 else 'are'} **{total}** {noun}:\n\n" + "\n".join(lines) + more
		+ "\n\nAsk me for the details of any of them."
	)


def _batch_display_names(doctype: str, names: list[str]) -> dict[str, str]:
	"""Map record names to their display titles in a single query."""
	if not names:
		return {}
	meta = frappe.get_meta(doctype)
	title_field = None
	for fieldname in (meta.title_field, f"{frappe.scrub(doctype)}_name"):
		if fieldname and meta.has_field(fieldname):
			title_field = fieldname
			break
	if not title_field:
		return {n: n for n in names}
	rows = frappe.get_list(
		doctype,
		filters={"name": ["in", names]},
		fields=["name", title_field],
		limit=len(names),
	)
	result: dict[str, str] = {}
	for r in rows:
		val = r.get(title_field)
		if isinstance(val, str) and val.strip():
			result[r["name"]] = val.strip()[:80]
		else:
			result[r["name"]] = r["name"]
	return {n: result.get(n, n) for n in names}


def _display_name(doctype: str, name: str) -> str:
	return _batch_display_names(doctype, [name]).get(name, name)


KNOWLEDGE_SEARCH_SLUG = "search_knowledge"

_KNOWLEDGE_SEARCH_DESCRIPTION = """Search this agent's knowledge bases for passages relevant to `query`.

Use this to ground answers in the agent's curated knowledge before relying on your own. Returns the \
most relevant chunks, each with its text, similarity score, and source. The knowledge bases searched \
are fixed by the agent's configuration — you cannot choose, add, or widen them."""


def bind_search_knowledge(kbs: list[str]) -> Tool:
	"""Build a `search_knowledge` tool scoped to `kbs`. The model sees only `query`; the
	knowledge bases come from the agent's config and cannot be chosen or widened. The
	registered builtin binds an empty list, so an unbound call fails closed in `retrieve`."""

	def search_knowledge(query: str) -> list[dict[str, Any]]:
		from flow.knowledge.retriever import retrieve

		return retrieve(query, kbs=kbs)

	return tool(search_knowledge, description=_knowledge_search_description(kbs))


def _knowledge_search_description(kbs: list[str]) -> str:
	"""Append the bound knowledge bases' descriptions so the model knows what's searchable
	and when to call the tool."""
	if not kbs:
		return _KNOWLEDGE_SEARCH_DESCRIPTION
	rows = frappe.get_all(
		"Flow Knowledge Base",
		filters={"name": ["in", kbs], "enabled": 1},
		fields=["title", "description"],
	)
	listed = "\n".join(f"- {r.title}: {r.description}" for r in rows if r.description)
	if not listed:
		return _KNOWLEDGE_SEARCH_DESCRIPTION
	return f"{_KNOWLEDGE_SEARCH_DESCRIPTION}\n\nThis agent's knowledge bases:\n{listed}"


search_knowledge = bind_search_knowledge([])


_UPDATE_MEMORY_DESCRIPTION = """Save a durable fact to persistent memory, or edit one by passing its memory_id.

Saved memories appear in the <agent_memory> block of your system prompt on every turn, \
including future conversations.

When to save: stable, reusable facts learned during the conversation — mappings and \
identifiers (e.g. an invoice item name to its ERP item code), business rules, corrections \
the user gives you, and their preferences. Do not save transient conversation state, \
secrets or credentials, or anything you can re-derive by reading records.

How to write: one short, self-contained, third-person fact per memory. Before adding, \
check <agent_memory> — if a related memory exists, pass its memory_id to revise or extend \
it instead of adding a duplicate. When a fact changes, edit the existing memory to the new \
value. Near the memory limit, consolidate related memories into one.

scope:
- "agent" — true for everyone who uses this agent (mappings, business rules, conventions).
- "user" — specific to the current user (their preferences and defaults).
Ask: is this about the organisation, or about this person?

keywords: optional space-separated search terms that help this memory resurface later — \
synonyms, alternate names, codes, or the words a user would ask with (e.g. for a fact about \
stationery tax: "pens paper pencils office supplies GST"). They are used only for retrieval, \
never shown as part of the fact. Add them when the fact's wording differs from how it will be \
asked about."""


def bind_update_memory(agent: str | None) -> Tool:
	"""Build an `update_memory` tool bound to `agent`. The binding comes from the agent's
	config, never the model. The registered builtin binds None, so an unbound call
	fails closed."""

	def update_memory(
		content: str,
		scope: Literal["agent", "user"],
		memory_id: str | None = None,
		keywords: str | None = None,
	) -> dict[str, Any]:
		from flow.memory.memory import save_memory

		if not agent:
			frappe.throw(_("Memory is not configured for this agent."), title=_("Memory Unavailable"))
		return save_memory(agent, content=content, scope=scope, memory_id=memory_id, keywords=keywords)

	return tool(update_memory, description=_UPDATE_MEMORY_DESCRIPTION)


update_memory = bind_update_memory(None)


@tool(
	requires_confirmation=True,
	confirm_prompt=lambda args: (
		f"{args.get('description') or _('Run Python code')}:\n\n{args.get('code', '')}"
	),
)
def execute(code: str, description: str) -> Any:
	"""Run Python in a permission-respecting sandbox for computation, emails, or multi-record work.

	`description` is one short, plain-English sentence stating what this code does, for a
	non-technical user who approves it — e.g. "Count open ToDos". Describe the intent, not the code.

	Do NOT write `import` statements — imports are blocked and the whole script fails. `frappe`
	and `frappe.utils` are already in scope; everything you can use is listed below, so never
	start with `import ...`.

	Every function here enforces the current user's permissions — there is no way to read or
	write data the user cannot access. Assign the value to return to a variable named `result`.
	Example (no imports, just use `frappe` directly):
	    result = frappe.db.count("ToDo", {"status": "Open"})

	Available:
	- Reads: frappe.get_list (supports group_by and aggregates via dict fields, e.g.
	  fields=[{"SUM": "qty", "as": "total"}] or [{"COUNT": "*", "as": "n"}]),
	  frappe.get_doc (returns a dict), frappe.get_meta, frappe.db.get_value/get_single_value/count/exists.
	- Writes: create, update, delete, run_action — the same permission-checked tools you call directly.
	- Also: read, describe, find_doctypes, frappe.call (whitelisted methods), frappe.enqueue,
	  frappe.sendmail, frappe.get_print, frappe.utils.* (dates, numbers, strings).

	Sandbox limits — code using these FAILS:
	- No `import`. `frappe` and `frappe.utils` are already in scope; nothing else can be imported.
	- No names or attributes starting with `_` (no dunders, no `obj._private`).
	- No raw database access: frappe.db.sql, frappe.qb, frappe.db.set_value and frappe.get_all are
	  unavailable — use frappe.get_list and the write tools, which respect permissions.
	- Unavailable builtins: open, eval, exec, compile, getattr, setattr, hasattr,
	  globals, locals, vars, dir, type, input. Available: len, range, str, int, float,
	  bool, sum, sorted, enumerate, zip, min, max, abs, dict, list, set, tuple.
	- `str.format()` / `.format_map()` are blocked — use f-strings or `%` formatting.
	- `print()` output is logged, not returned — put what you want back into `result`.

	The user approves each call before it runs.
	"""
	exec_globals, _locals = safe_exec(code, script_filename="ai_execute")
	return exec_globals.get("result")


def _error_text(e: Exception) -> str:
	"""Some frappe exceptions carry their message in the message log, not str() — fall back to the type."""
	return (str(e).strip() or e.__class__.__name__)[:_ERROR_LIMIT]


def _summarize_names(names: list[str] | None, limit: int = 6) -> str:
	names = names or []
	shown = ", ".join(str(n) for n in names[:limit])
	if len(names) > limit:
		shown += f" … +{len(names) - limit} more"
	return shown or "—"


def _doc_actions(doc: Any, meta: Any) -> dict[str, Any]:
	"""Actions the current user can run on this record: lifecycle, workflow, methods."""
	lifecycle: list[str] = []
	if getattr(meta, "is_submittable", 0):
		lifecycle.append(_LIFECYCLE_BY_DOCSTATUS.get(int(doc.docstatus)))
	if int(doc.docstatus) != 1 and frappe.has_permission(doc.doctype, "delete", doc.name):
		lifecycle.append("delete")
	if getattr(meta, "allow_rename", 0):
		lifecycle.append("rename")
	return {
		"lifecycle": [a for a in lifecycle if a],
		"workflow": sorted(_workflow_actions(doc)),
		"methods": _whitelisted_methods(doc.doctype),
	}


def _workflow_actions(doc: Any) -> set[str]:
	from frappe.model.workflow import get_transitions, get_workflow_name

	if not get_workflow_name(doc.doctype):
		return set()
	try:
		return {t.get("action") for t in get_transitions(doc) if t.get("action")}
	except Exception:
		return set()


def _whitelisted_methods(doctype: str) -> list[str]:
	"""Custom whitelisted controller methods (the app-specific form buttons), excluding base Document methods."""
	from frappe.model.base_document import get_controller
	from frappe.model.document import Document

	try:
		controller = get_controller(doctype)
	except Exception:
		return []
	base = set(dir(Document))
	methods = set()
	for attr_name in dir(controller):
		if attr_name.startswith("_") or attr_name in base:
			continue
		attr = getattr(controller, attr_name, None)
		if callable(attr) and getattr(attr, "__func__", attr) in frappe.whitelisted:
			methods.add(attr_name)
	return sorted(methods)


def _resolve_method(doc: Any, action: str) -> Any:
	method = getattr(doc, action, None)
	if callable(method) and getattr(method, "__func__", method) in frappe.whitelisted:
		return method
	return None


def _apply_action(doctype: str, name: str, action: str, args: dict[str, Any]) -> Any:
	doc = frappe.get_doc(doctype, name)
	if action == "submit":
		doc.submit()
		return {"name": doc.name, "docstatus": int(doc.docstatus)}
	if action == "cancel":
		doc.cancel()
		return {"name": doc.name, "docstatus": int(doc.docstatus)}
	if action == "amend":
		amended = frappe.copy_doc(doc)
		amended.amended_from = doc.name
		amended.insert()
		return {"name": amended.name}
	if action in _workflow_actions(doc):
		from frappe.model.workflow import apply_workflow

		apply_workflow(doc, action)
		return {"name": doc.name, "action": action}
	if _resolve_method(doc, action) is not None:
		return doc.run_method(action, **args)
	raise ValueError(f"Unknown action {action!r} for {doctype}")


def _precheck_create(args: dict[str, Any], context: str | None = None) -> str | None:
	# A single object plainly means one record — small models often drop the list brackets.
	# Normalised in place so the approval card and the call itself both see a list.
	if isinstance(args.get("records"), dict):
		args["records"] = [args["records"]]
	records = args.get("records")
	if isinstance(records, list):
		args["records"] = records = [_normalize_values(args.get("doctype"), r) for r in records]
	if not isinstance(records, list) or not records or not all(isinstance(r, dict) and r for r in records):
		return (
			"records must be a non-empty list of field-value objects. describe() the DocType for its "
			"required fields and ask the user for any values you don't have before calling create."
		)
	for row, values in enumerate(records):
		try:
			misplaced = _misplaced_child_fields(args.get("doctype") or "", values)
		except Exception:
			return None  # unknown doctype etc.: let the tool itself report it
		if misplaced:
			return f"Row {row}: {misplaced}"
	doctype = args.get("doctype")
	return (
		_unknown_fields_error(doctype, records)
		or _missing_required_error(doctype, records)
		or _ungrounded_error(doctype, records, context)
		or _resolve_links_error(doctype, records)
	)


def _normalize_values(doctype: str | None, values: Any) -> Any:
	"""Fix the two ways models most often mis-key a record: field labels instead of fieldnames
	("Date of Birth" -> date_of_birth) and dates as the user typed them in the site's format
	("01-06-2003" on a dd-mm-yyyy site -> 2003-06-01; Frappe itself would read it month-first).
	Child-table rows are normalised against their own DocType."""
	if not isinstance(values, dict) or not doctype or not frappe.db.exists("DocType", doctype):
		return values
	meta = frappe.get_meta(doctype)
	# Data fields only: a Section Break can share a label with its table ("Items").
	by_label = {
		(f.label or "").strip().lower(): f.fieldname
		for f in meta.fields
		if f.label and f.fieldtype not in LAYOUT_FIELDTYPES
	}
	out: dict[str, Any] = {}
	for key, value in values.items():
		fieldname = key if meta.has_field(key) else by_label.get(str(key).strip().lower(), key)
		if not meta.has_field(fieldname):
			# An unknown key whose value fits exactly one unused field ("user_name": an email,
			# on User whose only email field is `email`) is that field.
			fits = [
				f.fieldname
				for f in meta.fields
				if _value_fits(f, value) and f.fieldname not in values and f.fieldname not in out
			]
			if len(fits) == 1:
				fieldname = fits[0]
		field = meta.get_field(fieldname)
		if field and field.fieldtype in ("Table", "Table MultiSelect") and isinstance(value, list):
			value = [_normalize_values(field.options, row) for row in value]
		elif field and field.fieldtype == "Date" and isinstance(value, str):
			value = _to_iso_date(value)
		elif field and field.fieldtype in NUMERIC_FIELDTYPES and isinstance(value, str):
			value = _to_number(value, int if field.fieldtype == "Int" else float)
		out[fieldname] = value
	return out


NUMERIC_FIELDTYPES = frozenset({"Int", "Float", "Currency", "Percent"})


def _to_number(value: str, kind: type) -> Any:
	"""A number typed as text ("100", "1,500.50") as a number — ERPNext's amount maths fails on
	strings (abs() of a str). Anything that isn't a number is returned for Frappe to reject."""
	try:
		return kind(float(value.replace(",", "").strip()))
	except ValueError:
		return value


SITE_DATE_DIRECTIVES = {"dd": "%d", "mm": "%m", "yyyy": "%Y"}


def _to_iso_date(value: str) -> str:
	"""A date typed in the site's format (any of - / . as separator) as YYYY-MM-DD; other
	values are returned unchanged for Frappe to validate."""
	from datetime import datetime

	text = value.strip()
	if DATE_PATTERN.match(text):
		return text
	site_format = (frappe.db.get_single_value("System Settings", "date_format") or "").lower()
	parts = re.split(r"[-/.]", site_format)
	if sorted(parts) != sorted(SITE_DATE_DIRECTIVES):
		return value
	for sep in ("-", "/", "."):
		try:
			pattern = sep.join(SITE_DATE_DIRECTIVES[p] for p in parts)
			return datetime.strptime(text, pattern).date().isoformat()
		except ValueError:
			continue
	return value


def _unknown_fields_error(doctype: str | None, records: list[dict[str, Any]]) -> str | None:
	"""Name keys that are not fields (after label mapping). Frappe would drop them silently —
	a `quantity` meant for `qty` just vanishes from the saved record."""
	if not doctype or not frappe.db.exists("DocType", doctype):
		return None
	problems = []
	for record in records:
		problems += _unknown_in(doctype, record)
	if not problems:
		return None
	return "Unknown fields: " + "; ".join(problems) + ". Use the fieldnames from describe, then call again."


def _unknown_in(doctype: str, values: dict[str, Any]) -> list[str]:
	import difflib

	meta = frappe.get_meta(doctype)
	problems = []
	for key, value in values.items():
		if key in _STANDARD_KEYS:
			continue
		field = meta.get_field(key)
		if field is None:
			names = [f.fieldname for f in meta.fields if f.fieldtype not in LAYOUT_FIELDTYPES]
			# Fields that fit the value come first: an email address under "user_name" is meant
			# for User's Email, not Username.
			fits = [f.fieldname for f in meta.fields if _value_fits(f, value)]
			close = difflib.get_close_matches(str(key).lower(), names, n=3, cutoff=0.5)
			close = list(dict.fromkeys(fits[:2] + close))[:3]
			hint = f" (did you mean {', '.join(close)}?)" if close else ""
			problems.append(f"{key!r} is not a field of {doctype}{hint}")
		elif field.fieldtype in ("Table", "Table MultiSelect") and isinstance(value, list):
			for row in value:
				if isinstance(row, dict):
					problems += _unknown_in(field.options, row)
	return problems


EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_VALUE = re.compile(r"^\+?[\d\s()-]{7,}$")


def _value_fits(field: Any, value: Any) -> bool:
	"""Whether a value's shape plainly matches a field's type (email, phone, date)."""
	if not isinstance(value, str) or field.fieldtype in LAYOUT_FIELDTYPES:
		return False
	text = value.strip()
	if EMAIL_VALUE.match(text):
		return field.fieldtype == "Data" and (field.options == "Email" or "email" in field.fieldname)
	if PHONE_VALUE.match(text):
		return field.fieldtype == "Data" and (
			field.options == "Phone" or field.fieldname in ("phone", "mobile_no", "cell_number")
		)
	if DATE_PATTERN.match(text):
		return field.fieldtype == "Date"
	return False


def _resolve_links_error(doctype: str | None, records: list[dict[str, Any]]) -> str | None:
	"""Swap a Link value that is a record's display name for its ID (an item's name for its item
	code), in place. Errors when no record, or more than one, matches."""
	if not doctype or not frappe.db.exists("DocType", doctype):
		return None
	problems = []
	for record in records:
		problems += _resolve_links_in(doctype, record)
	if not problems:
		return None
	return "; ".join(problems) + ". Look the record up with read or ask the user, then call again."


def _resolve_links_in(doctype: str, values: dict[str, Any]) -> list[str]:
	meta = frappe.get_meta(doctype)
	problems = []
	for key, value in list(values.items()):
		field = meta.get_field(key)
		if field is None:
			continue
		if field.fieldtype in ("Table", "Table MultiSelect") and isinstance(value, list):
			for row in value:
				if isinstance(row, dict):
					problems += _resolve_links_in(field.options, row)
			continue
		if not isinstance(value, str) or not value:
			continue
		if field.fieldtype == "Link":
			target = field.options
		elif field.fieldtype == "Dynamic Link":
			target = values.get(field.options)  # the DocType chosen in the companion field
		else:
			continue
		if not target or target == "DocType" or not frappe.db.exists("DocType", target):
			continue
		if frappe.db.exists(target, {"name": value}):
			continue
		matches = _records_titled(target, value)
		if len(matches) == 1:
			values[key] = matches[0]
		elif matches:
			problems.append(f"{field.label} {value!r} matches several {target} records: {', '.join(matches)}")
		else:
			problems.append(f"No {target} with ID or name {value!r} (field {field.label})")
	return problems


def _records_titled(doctype: str, title: str) -> list[str]:
	"""IDs of the records whose display name equals `title` (case-insensitive)."""
	meta = frappe.get_meta(doctype)
	candidates = [meta.title_field, f"{frappe.scrub(doctype)}_name", "title", "full_name"]
	for fieldname in dict.fromkeys(c for c in candidates if c and meta.has_field(c)):
		names = frappe.get_list(doctype, filters={fieldname: title}, pluck="name", limit=5)
		if names:
			return names
	return []


def _missing_required_error(doctype: str | None, records: list[dict[str, Any]]) -> str | None:
	"""Name the required fields a record lacks, so the model asks the user for them instead of
	putting an incomplete record in front of them for approval (and failing on save)."""
	if not doctype or not frappe.db.exists("DocType", doctype):
		return None
	meta = frappe.get_meta(doctype)
	_defaults, sole_links = _field_facts(doctype)
	required = [f for f in _user_filled(meta) if f.fieldname not in sole_links]
	for row, values in enumerate(records):
		for fieldname, only in sole_links.items():
			if meta.has_field(fieldname) and not values.get(fieldname):
				values[fieldname] = only
		missing = [f.label or f.fieldname for f in required if values.get(f.fieldname) in (None, "")]
		if missing:
			return (
				f"Row {row} is missing required fields: {', '.join(missing)}. Ask the user for these "
				"values (all of them in one message), then call create again."
			)
	return None


def _precheck_update(args: dict[str, Any], context: str | None = None) -> str | None:
	if not args.get("values"):
		return "values must be a non-empty object of the fields to change."
	if isinstance(args["values"], dict):
		args["values"] = _normalize_values(args.get("doctype"), args["values"])
	doctype, values = args.get("doctype"), [args["values"]]
	return (
		_precheck_names(args)
		or _unknown_fields_error(doctype, values)
		or _ungrounded_error(doctype, values, context)
		or _resolve_links_error(doctype, values)
	)


def _update_to_create(args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
	"""An update whose records all don't exist, carrying values, is a request to make a new
	record (a "create a user" request sent as an update of a User ID that doesn't exist): run it as a create, which
	applies its own checks and asks the user to approve."""
	doctype, names, values = args.get("doctype"), args.get("names"), args.get("values")
	if not (isinstance(values, dict) and values and isinstance(names, list) and names):
		return None
	if not doctype or not frappe.db.exists("DocType", doctype):
		return None
	if any(frappe.db.exists(doctype, {"name": n}) or _records_titled(doctype, n) for n in names if isinstance(n, str)):
		return None
	# Only a complete new record: "close the todo X" (status only) is an update of a record
	# the model named loosely, not a request to make one.
	record = _normalize_values(doctype, dict(values))
	if _missing_required_error(doctype, [record]):
		return None
	return "create", {"doctype": doctype, "records": [dict(values)]}


def _precheck_names(args: dict[str, Any], context: str | None = None) -> str | None:
	names = args.get("names")
	if not isinstance(names, list) or not names:
		return (
			"names must be a non-empty list of existing record IDs (from count or read). "
			"To make a NEW record, call create instead."
		)
	doctype = args.get("doctype") or ""
	if not frappe.db.exists("DocType", doctype):
		return None  # let the tool report the unknown DocType
	# A record named by its title (an employee's name for their ID) resolves to its ID when
	# exactly one record matches.
	resolved = []
	for n in names:
		if isinstance(n, str) and not frappe.db.exists(doctype, {"name": n}):
			matches = _records_titled(doctype, n)
			n = matches[0] if len(matches) == 1 else n
		resolved.append(n)
	args["names"] = names = resolved
	missing = [n for n in names if not frappe.db.exists(doctype, {"name": n})]
	if missing:
		return (
			f"No {doctype} record with ID {', '.join(map(repr, missing))}. This tool only works on "
			"existing records. To make a NEW record, call create(doctype, records=[{...}]) with the "
			"values the user gave; to find existing IDs, use count or read."
		)
	return None


def _precheck_run_action(args: dict[str, Any], context: str | None = None) -> str | None:
	problem = _precheck_names(args, context)
	if problem:
		return problem
	doctype, action = args["doctype"], args.get("action")
	if not frappe.db.exists("DocType", doctype):
		return None
	meta = frappe.get_meta(doctype)
	doc = frappe.get_doc(doctype, args["names"][0])
	available = _doc_actions(doc, meta)
	valid = [
		a for a in available["lifecycle"] if a in ("submit", "cancel", "amend")
	] + available["workflow"] + available["methods"]
	if action not in valid:
		listed = ", ".join(valid) or "none"
		return f"{action!r} is not an available action for {doctype} {doc.name}. Available: {listed}."
	return None


def _ungrounded_error(doctype: str | None, records: list[Any], context: str | None) -> str | None:
	"""Reject values that appear nowhere in what the user said or a tool returned — i.e. that
	the model made up (a sample "John Smith" employee). A field's own default is allowed.
	Skipped when `context` is None: the user has already seen and approved the values."""
	if context is None:
		return None
	defaults, sole_links = _field_facts(doctype)
	invented = sorted(
		{
			f"{key}={value}"
			for record in records
			for key, value in _leaf_values(record)
			if not _is_grounded(value, context)
			and str(value).lower() != defaults.get(key)
			and value != sole_links.get(key)
		}
	)
	if not invented:
		return None
	date_format = frappe.db.get_single_value("System Settings", "date_format") or "yyyy-mm-dd"
	return (
		f"These values were not given by the user or found by a tool: {', '.join(invented)}. "
		"Never make up or guess data — ask the user for these values, then call again. "
		f"Write dates as YYYY-MM-DD; the user types dates as {date_format}."
	)


def _leaf_values(value: Any, key: str = "") -> list[tuple[str, Any]]:
	"""(fieldname, scalar) pairs in a record, descending into child-table rows."""
	if isinstance(value, dict):
		return [pair for k, v in value.items() for pair in _leaf_values(v, k)]
	if isinstance(value, list):
		return [pair for v in value for pair in _leaf_values(v, key)]
	if value is None or isinstance(value, bool) or key in ("doctype", "name"):
		return []
	return [(key, value)]


def _is_grounded(value: Any, context: str) -> bool:
	if isinstance(value, int | float):
		number = int(value) if float(value).is_integer() else value
		return _mentions(context, str(number))
	text = str(value).strip().lower()
	if not text:
		return True
	if _mentions(context, text):
		return True
	if DATE_PATTERN.match(text):
		return any(_mentions(context, form) for form in _date_forms(text[:10]))
	# Built from what the user typed (an initial written "K." when they typed "k"): every word must appear.
	words = WORD_PATTERN.findall(text)
	return bool(words) and all(_mentions(context, w) for w in words if len(w) > 1)


def _mentions(context: str, text: str) -> bool:
	"""`text` occurs in `context` as a whole token, not inside a longer word or number."""
	return re.search(rf"(?<![\w]){re.escape(text)}(?![\w])", context) is not None


def _date_forms(iso: str) -> set[str]:
	"""Ways a person may have typed an ISO date: ISO itself, the site's format, and common
	day-first / month-first / written forms ("1 sep 2026"). Day and month must both match —
	the year alone let a made-up 2026-01-01 pass for "01-09-2026"."""
	try:
		date = frappe.utils.getdate(iso)
	except Exception:
		return {iso}
	d, m, y = date.day, date.month, date.year
	# Numeric dates are read in the site's order only: "01-02-2002" is 1 Feb on a dd-mm-yyyy
	# site, and must not also vouch for 2 Jan.
	site_format = (frappe.db.get_single_value("System Settings", "date_format") or "dd-mm-yyyy").lower()
	first, second = (m, d) if site_format.startswith("mm") else (d, m)
	forms = {iso}
	for sep in ("-", "/", "."):
		forms |= {f"{first:02d}{sep}{second:02d}{sep}{y}", f"{first}{sep}{second}{sep}{y}"}
	for month in (date.strftime("%b").lower(), date.strftime("%B").lower()):
		forms |= {f"{d} {month} {y}", f"{d:02d} {month} {y}", f"{month} {d} {y}", f"{month} {d}, {y}"}
	forms.add(frappe.utils.formatdate(iso, site_format).lower())
	return forms


def _field_facts(doctype: str | None) -> tuple[dict[str, str], dict[str, str]]:
	"""Values allowed without the user saying them, across the DocType and its child tables:
	(fieldname -> lower-cased default, fieldname -> the only record a Link field can hold,
	e.g. the site's single Company)."""
	if not doctype or not frappe.db.exists("DocType", doctype):
		return {}, {}
	cache = frappe.flags.flow_field_facts = getattr(frappe.flags, "flow_field_facts", None) or {}
	if doctype in cache:
		return cache[doctype]
	meta = frappe.get_meta(doctype)
	fields = [f for m in [meta, *(frappe.get_meta(t.options) for t in meta.get_table_fields())] for f in m.fields]
	defaults = {f.fieldname: str(f.default).lower() for f in fields if f.default}
	sole_links: dict[str, str] = {}
	for f in fields:
		if f.fieldtype == "Link" and f.reqd and f.options:
			names = frappe.get_all(f.options, pluck="name", limit=2)
			if len(names) == 1:
				sole_links[f.fieldname] = names[0]
	cache[doctype] = (defaults, sole_links)
	return cache[doctype]


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
# Dots only inside a word, so an email address stays one token but an initial "K." matches "k".
WORD_PATTERN = re.compile(r"[\w@+-]+(?:\.[\w@+-]+)*")


@tool(
	requires_confirmation=True,
	precheck=_precheck_create,
	route=lambda text: _route_create(text),
	confirm_prompt=lambda args: (
		_("Create {0} {1} record(s):\n\n{2}").format(
			len(args.get("records") or []),
			args.get("doctype", "?"),
			_summarize_values((args.get("records") or [{}])[0]),
		)
	),
)
def create(
	doctype: DocTypeArg,
	records: Annotated[
		list[dict[str, Any]],
		'One object per record, by field. Item rows go in their table: {"items": [{"item_code": ..., "qty": ..., "rate": ...}]}.',
	],
) -> dict[str, Any]:
	"""Create records with values the user gave. The user approves first."""
	if not frappe.has_permission(doctype, "create"):
		raise PermissionError(f"No permission to create {doctype}")

	created: list[str] = []
	failures: list[dict[str, Any]] = []
	for row, values in enumerate(records):
		misplaced = _misplaced_child_fields(doctype, values or {})
		if misplaced:
			failures.append({"row": row, "error": misplaced})
			continue
		try:
			doc = frappe.new_doc(doctype)
			doc.update(values or {})
			doc.insert()
			created.append(doc.name)
		except Exception as e:
			failures.append({"row": row, "error": _error_text(e)})

	route = doctype.lower().replace(" ", "-")
	result: dict[str, Any] = {
		"doctype": doctype,
		"created": created,
		"links": [f"/desk/{route}/{quote(str(name))}" for name in created],
	}
	if failures:
		result["failures"] = failures
	return result


def _child_fields_map(doctype: str) -> list[tuple[str, str, set[str]]]:
	"""(table_fieldname, child_doctype, set_of_child_fieldnames) cached for the request."""
	cache = frappe.flags.flow_child_fields = getattr(frappe.flags, "flow_child_fields", None) or {}
	if doctype not in cache:
		meta = frappe.get_meta(doctype)
		res = []
		for table in meta.get_table_fields():
			child = frappe.get_meta(table.options)
			res.append((table.fieldname, table.options, {f.fieldname for f in child.fields}))
		cache[doctype] = res
	return cache[doctype]


def _misplaced_child_fields(doctype: str, values: dict[str, Any]) -> str | None:
	"""Explain top-level keys that are really fields of one of the doctype's child tables
	(e.g. `item_code` on a Sales Invoice). Frappe would otherwise drop them silently and fail
	later with an error that says nothing about the cause (an invoice with no items crashes
	computing totals), leaving the model nothing to correct."""
	meta = frappe.get_meta(doctype)
	unknown = [key for key in values if not meta.has_field(key) and key not in _STANDARD_KEYS]
	if not unknown:
		return None

	hints: list[str] = []
	for table_fieldname, options, child_fieldnames in _child_fields_map(doctype):
		owned = [key for key in unknown if key in child_fieldnames]
		if owned:
			example = ", ".join(f'"{key}": ...' for key in owned)
			hints.append(
				f"{', '.join(owned)} belong to the child table `{table_fieldname}` ({options}); "
				f'pass them as "{table_fieldname}": [{{{example}}}]'
			)
			unknown = [key for key in unknown if key not in owned]
	if not hints:
		return None
	return f"Fields not on {doctype}: " + "; ".join(hints) + ". Nothing was created for this row."


_STANDARD_KEYS = frozenset({"doctype", "name", "__newname"})


@tool(
	requires_confirmation=True,
	precheck=_precheck_update,
	redirect=_update_to_create,
	confirm_prompt=lambda args: (
		_("Update {0} {1} ({2}):\n\n{3}").format(
			len(args.get("names") or []),
			args.get("doctype", "?"),
			_summarize_names(args.get("names")),
			_summarize_values(args.get("values")),
		)
	),
)
def update(
	doctype: DocTypeArg,
	names: NamesArg,
	values: Annotated[
		dict[str, Any], 'Only the fields to change, by fieldname, e.g. {"status": "Closed"}.'
	],
) -> dict[str, Any]:
	"""Change fields on existing records (the same values on each). Returns {updated, failures}.
	The user approves first. Submitted records can't be edited — cancel and amend via run_action."""
	updated: list[str] = []
	failures: list[dict[str, Any]] = []
	for name in names:
		try:
			if not frappe.has_permission(doctype, "write", name):
				raise frappe.PermissionError(_("No permission to update {0} {1}.").format(doctype, name))
			doc = frappe.get_doc(doctype, name)
			doc.update(values or {})
			doc.save()
			updated.append(doc.name)
		except Exception as e:
			failures.append({"name": name, "error": _error_text(e)})

	result: dict[str, Any] = {"doctype": doctype, "updated": updated}
	if failures:
		result["failures"] = failures
	return result


@tool(
	requires_confirmation=True,
	precheck=_precheck_names,
	confirm_prompt=lambda args: (
		_("Delete {0} {1}: {2}").format(
			len(args.get("names") or []),
			args.get("doctype", "?"),
			_summarize_names(args.get("names")),
		)
	),
)
def delete(doctype: DocTypeArg, names: NamesArg) -> dict[str, Any]:
	"""Delete records. Returns {deleted, failures}; a record fails if others link to it or it is
	submitted (cancel it first). The user approves first."""
	deleted: list[str] = []
	failures: list[dict[str, Any]] = []
	for name in names:
		try:
			if not frappe.has_permission(doctype, "delete", name):
				raise frappe.PermissionError(_("No permission to delete {0} {1}.").format(doctype, name))
			frappe.delete_doc(doctype, name, ignore_missing=False)
			deleted.append(name)
		except Exception as e:
			failures.append({"name": name, "error": _error_text(e)})

	result: dict[str, Any] = {"doctype": doctype, "deleted": deleted}
	if failures:
		result["failures"] = failures
	return result


@tool(
	requires_confirmation=True,
	precheck=_precheck_run_action,
	confirm_prompt=lambda args: (
		_("Run '{0}' on {1} {2}: {3}").format(
			args.get("action"),
			len(args.get("names") or []),
			args.get("doctype", "?"),
			_summarize_names(args.get("names")),
		)
	),
)
def run_action(
	doctype: DocTypeArg,
	names: NamesArg,
	action: Annotated[
		str,
		'One of the actions describe(doctype, name) lists, e.g. "submit", "cancel", "amend", '
		"or a workflow transition. Never invent one.",
	],
	args: Annotated[dict[str, Any] | None, "Arguments for a method action; usually omit."] = None,
) -> dict[str, Any]:
	"""Run a document action (submit, cancel, amend, rename, workflow transition, or a
	whitelisted method) on records. The user approves first."""
	args = args or {}

	if action == "rename":
		if len(names) != 1:
			raise ValueError("rename acts on a single document; pass exactly one name.")
		new_name = args.get("new_name")
		if not new_name:
			raise ValueError("rename requires args.new_name.")
		return {"action": "rename", "old": names[0], "new": frappe.rename_doc(doctype, names[0], new_name)}

	results: list[dict[str, Any]] = []
	failures: list[dict[str, Any]] = []
	for name in names:
		try:
			results.append({"name": name, "result": _apply_action(doctype, name, action, args)})
		except Exception as e:
			failures.append({"name": name, "error": _error_text(e)})

	result: dict[str, Any] = {"action": action, "results": results}
	if failures:
		result["failures"] = failures
	return result


from flow.tools.answers import (  # noqa: E402
	assignment_steps,
	document_flow,
	explain_doctype,
	role_permissions,
	show_records,
	status_steps,
)
from flow.tools.diagnosis import error_diagnosis  # noqa: E402
from flow.tools.web import docs_howto, frappe_docs, web_search  # noqa: E402

BUILTIN_TOOLS: list[Tool] = [
	find_doctypes,
	describe,
	creation_steps,
	required_values,
	small_talk,
	error_diagnosis,
	show_records,
	explain_doctype,
	document_flow,
	assignment_steps,
	status_steps,
	role_permissions,
	frappe_docs,
	docs_howto,
	web_search,
	read,
	count,
	search_knowledge,
	update_memory,
	create,
	update,
	delete,
	run_action,
	execute,
]


def sync_builtin_tools() -> None:
	"""Upsert builtin tools as Flow Tool rows. Uses db.set_value to bypass the immutability
	guard in FlowTool.validate (which protects user edits, not system migration)."""
	for builtin in BUILTIN_TOOLS:
		import_path = f"flow.tools.builtins.{builtin.name}"
		if frappe.db.exists("Flow Tool", builtin.name):
			frappe.db.set_value(
				"Flow Tool",
				builtin.name,
				{
					"import_path": import_path,
					"description": builtin.description,
					"requires_confirmation": int(builtin.requires_confirmation),
					"is_system_generated": 1,
				},
			)
		else:
			frappe.get_doc(
				{
					"doctype": "Flow Tool",
					"slug": builtin.name,
					"title": builtin.name.replace("_", " ").title(),
					"type": "Imported",
					"import_path": import_path,
					"description": builtin.description,
					"is_system_generated": 1,
					"requires_confirmation": int(builtin.requires_confirmation),
				}
			).insert(ignore_permissions=True)
