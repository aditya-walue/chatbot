# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""error_diagnosis: explain an error message from facts on this site, not from memory.

Sources, each optional:
- the traceback's own frames in the installed apps (file, line, function), when one was pasted;
- the code that raises the message, found by searching the installed apps' source (exact text
  first, then its fixed parts, so templated messages like "Could not find {0}" still match);
- scripts and templates saved on the site, recent Error Log entries and other failure logs;
- a table of common Frappe / ERPNext error types with their usual cause and fix;
- when none of those knows the message, the Frappe documentation, then the web (the fixed start
  of the message only, never the values in it).

The result is a finished, formatted diagnosis shown to the user as-is (a final-answer tool).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Annotated, Any
from urllib.parse import quote

import frappe

from flow.lib.tool import LayaRoute, tool

SOURCE_EXTENSIONS = (".py", ".js", ".vue", ".ts")
# Generated bundles, dependencies and tests: matches there are noise, not where an error is raised.
SKIPPED_DIRS = frozenset(
	{"node_modules", "dist", ".git", "__pycache__", "tests", "test", "locale", "translations", "build", "fixtures"}
)
MAX_SOURCE_MATCHES = 2
MIN_FRAGMENT = 12
# Site scripts and print formats are searched on fragments only this long: a label such as "Delivery
# Date" is in many print formats and says nothing about where an error is raised.
MIN_SITE_FRAGMENT = 24
# This file quotes example messages in its docstrings, so a search of the app code would find it.
THIS_FILE = os.path.abspath(__file__)
# A pasted message may carry the user's own words around it ("getting this error <message> how
# to solve"): up to this many words are dropped from either end to find the message itself.
MAX_TRIMMED_WORDS = 4
MIN_WINDOW_WORDS = 5
# Longer lines are minified bundles, not readable source.
MAX_LINE = 400
MAX_ERROR_LOGS = 3
# Code and templates kept in the site's database rather than in an app: (DocType, fields searched).
SITE_CODE = (
	("Server Script", ("script",)),
	("Client Script", ("script",)),
	("Custom HTML Block", ("script", "html")),
	("Print Format", ("html",)),
	("Notification", ("message", "condition")),
)
# Other places a failure is recorded: (DocType, text field, filters, label).
FAILURE_LOGS = (
	("Scheduled Job Log", "details", {"status": "Failed"}, "Scheduled Job Log"),
	("Email Queue", "error", {"status": "Error"}, "Email Queue"),
	("Integration Request", "error", {"status": "Failed"}, "Integration Request"),
	("Webhook Request Log", "error", {}, "Webhook Request Log"),
)
# A traceback frame: File "/…/apps/<app>/<path>.py", line 12, in function
TRACE_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
MAX_FRAMES = 2
MIN_HELP_WORDS = 3
# Where a message's filled-in values begin: a quote, "Label: value", a bracket, "#1", a digit.
VALUE_START = re.compile(r"[\"']|:\s|\[|#|\d|<")
# Lines that raise or show a message rank above a mere mention of the text.
RAISE_MARKERS = ("throw", "msgprint", "raise", "_(", "__(", "show_alert", "ValidationError")

# A line that defines a label or a field ("label": _("Delivery Date"), a report column) holds the
# text but never raises it: it says nothing about where an error comes from.
DEFINITION_MARKERS = ('"label"', "'label'", "label:", "label=", '"fieldname"', "'fieldname'", "fieldtype", "columns")
# An old-style report column: _("Item") + ":Link/Item:120", "Qty:Float:80".
COLUMN_SPEC = re.compile(r"""[\w"']:(?:Link|Data|Date|Datetime|Float|Int|Currency|Percent|Check|Select|Time|Text)(?=[/:"'\d,])""")

# Pasted errors often carry the dialog's chrome; drop it before searching.
CHROME_PREFIXES = re.compile(
	r"^\s*(message|error|warning|alert|not permitted|validation error|traceback[^\n]*)\s*[:\n]\s*",
	re.IGNORECASE,
)
# The exception line of a Python traceback: "frappe.exceptions.LinkValidationError: Could not find …"
EXCEPTION_LINE = re.compile(r"^\s*(?:[\w.]+\.)?(\w+(?:Error|Exception)):\s*(.+)$", re.MULTILINE)

KNOWN_CAUSES: tuple[tuple[re.Pattern, str, str], ...] = tuple(
	(re.compile(pattern, re.IGNORECASE), cause, fix)
	for pattern, cause, fix in (
		(
			r"not permitted|no permission|insufficient permission|permissionerror|not allowed to",
			"Your role lacks permission for this document or action.",
			"Ask a System Manager to grant the permission in **Role Permission Manager** "
			"(or **User Permissions**), then reload.",
		),
		(
			r"is mandatory|mandatory fields? required|mandatoryerror|please (enter|select|set|specify)|cannot be (empty|blank)",
			"A required field is empty.",
			"Fill in the field named in the message, then save again.",
		),
		(
			r"could not find|does not exist|not found|linkvalidationerror",
			"A linked record named in the message doesn't exist (wrong name, deleted, or not yet created).",
			"Pick an existing record from the field's dropdown, or create the missing record first.",
		),
		(
			r"duplicate entry|already exists|duplicateentryerror|uniquevalidationerror",
			"A record with the same name or unique value already exists.",
			"Open the existing record instead, or change the name/value so it is unique.",
		),
		(
			r"cannot (edit|update|change).*(submitted|cancelled)|not allowed to change .* after submission|updateaftersubmiterror",
			"The document is submitted or cancelled, so it can no longer be edited.",
			"Cancel it and click **Amend** to make an editable copy, or create a new document.",
		),
		(
			r"cannot delete or cancel because .* is linked|linkexistserror|is linked with",
			"Other documents link to this one, so it can't be deleted or cancelled.",
			"Cancel or delete the linked documents listed in the message first.",
		),
		(
			r"fiscal year|accounting period|books (have been|are) closed|period closing",
			"The posting date falls in a closed or missing fiscal year / accounting period.",
			"Change the posting date, or ask an accountant to create or reopen the period.",
		),
		(
			r"naming series|series .* not set",
			"The document's naming series isn't set up.",
			"Set it in **Document Naming Settings** for this document type.",
		),
		(
			r"(document has been modified|timestampmismatch|has been modified after you have opened)",
			"Someone (or something) saved this document after you opened it.",
			"Reload the page, re-apply your changes and save again.",
		),
		(
			r"insufficient stock|negative stock|not enough stock|negativestockerror|needed in warehouse",
			"There isn't enough stock in the selected warehouse.",
			"Receive stock first (Purchase Receipt / Stock Entry), choose another warehouse, or reduce the quantity.",
		),
		(
			r"not a valid (phone|email|url)|invalid (phone|email|url)",
			"A phone number, email or URL isn't in a valid format.",
			"Correct the value's format: a phone number with its country code, or a full email address.",
		),
		(
			r"lock wait timeout|deadlock|queryTimeoutError|query execution was interrupted",
			"Two operations were changing the same data at once, or a query ran too long.",
			"Retry in a minute. If it keeps happening, avoid running big imports or reports at the same time.",
		),
		(
			r"data too long|integrityerror|datamismatch|incorrect (string|decimal|integer) value",
			"A value is too long, or doesn't fit the field's type or a database constraint.",
			"Shorten or correct the value in the field named in the message.",
		),
		(
			r"smtp|email account|outgoing email|authentication failed|could not send mail|default email account",
			"Outgoing email isn't set up correctly, or the mail server rejected the login.",
			"Check **Email Account** (outgoing) and its password or app password, then retry from **Email Queue**.",
		),
		(
			r"default (price list|warehouse|company|currency|cost center|account)|has no (account|default)",
			"A default the document needs (company, warehouse, price list, account...) is not set.",
			"Set it on the document, or as a default in the Company, Item, Stock Settings or Accounts Settings.",
		),
		(
			r"cannot (link|be linked to) (a )?cancelled|is cancelled|docstatus",
			"The document refers to another one that is cancelled, or is in a state that doesn't allow this action.",
			"Pick a valid (not cancelled) document, or amend the cancelled one.",
		),
		(
			r"too many requests|rate limit|429",
			"Too many requests were sent in a short time.",
			"Wait a minute and try again.",
		),
		(
			r"session expired|csrf|invalid request|login required|authenticationerror",
			"Your login session expired.",
			"Reload the page and log in again.",
		),
	)
)


# Unmistakably an error report: a traceback, an exception name, or a typical Frappe message.
ERROR_REPORT = re.compile(
	r"traceback \(most recent call last\)|\b\w+(error|exception)\b|^\s*(error|message)\s*[:\n]"
	r"|could not find|is mandatory|mandatory fields|not permitted|insufficient permission"
	r"|not correctly configured|please (check|enter|select|set)\b|does not exist|not allowed to"
	r"|cannot be (deleted|cancelled|empty)|duplicate entry|already exists|is not a valid"
	r"|session expired|negative stock|insufficient stock",
	re.IGNORECASE,
)
# Requests that mention an error but ask for something else ("how to ...", "create ...").
NOT_A_REPORT = re.compile(r"^\s*(how|create|add|make|show|list|count|update|delete|set)\b", re.IGNORECASE)


def _route_error(text: str) -> dict[str, Any] | None:
	if NOT_A_REPORT.match(text) or not ERROR_REPORT.search(text):
		return None
	return {"error": text}


@tool(
	final_answer=True,
	route=_route_error,
	laya=LayaRoute("TROUBLESHOOTING", lambda text, signals: {"error": text}),
)
def error_diagnosis(
	error: Annotated[str, "The error message exactly as the user pasted it, including any traceback."],
) -> dict[str, Any]:
	"""Diagnose any error or failure on this site, pasted or described ("why did X fail"): where it
	is raised (app code, scripts saved on the site), recent Error Log and failed jobs, emails
	and integration calls, the likely cause and the fix."""
	message, exception = _clean(error)
	if not message:
		raise ValueError("Paste the error message to diagnose.")

	frames = _traceback_frames(error) if _can_read_source() else []
	sources = _find_in_source(message) if _can_read_source() else []
	if not sources:
		# A message the user sees translated: search the code for its English original.
		sources = _find_translated_source(message)
	site_code = _find_in_site_code(message)
	logs = _recent_error_logs(message, exception)
	failures = _recent_failures(message)
	causes = [(cause, fix) for pattern, cause, fix in KNOWN_CAUSES if pattern.search(f"{exception} {message}")]

	if exception == "MandatoryError":
		missing = message.rsplit("]:", 1)[-1].strip()
		causes = [(f"Required field(s) left empty: {missing}.", "Fill in those fields, then save again.")]
	help_text = None
	if not (causes or sources or frames or logs or site_code or failures):
		help_text = _docs_help(message, exception)
	if not (causes or sources or frames or logs or site_code or failures or help_text):
		# Nothing on this site or in the docs explains it, so the text may not be an error report at
		# all (Laya also sends "I don't understand anything" here); no answer, so the chat model replies.
		# The model often repeats this sentence to the user, so it reads as something to say.
		return {
			"message": message,
			"answer_hint": "I couldn't find an explanation for this on your site or in the documentation. "
			"Tell me what you were doing when it appeared (which page, which button).",
		}
	return {"answer": _format(message, exception, sources, logs, causes[:2], site_code, failures, frames, help_text)}


def _traceback_frames(text: str) -> list[dict[str, Any]]:
	"""The innermost frames of a pasted traceback that are in an installed app (file relative to
	the apps folder, line, function): where it was raised, read from the paste itself."""
	installed = {app: _app_title(app) for app in frappe.get_installed_apps()}
	frames = []
	for path, line, function in TRACE_FRAME.findall(text or ""):
		found = re.search(r"/apps/([^/]+)/(.+)$", path.replace(os.sep, "/"))
		if found and found.group(1) in installed:
			frames.append({"app": installed[found.group(1)], "file": f"{found.group(1)}/{found.group(2)}", "line": int(line), "function": function})
	return frames[-MAX_FRAMES:][::-1]


def _app_title(app: str) -> str:
	return (frappe.get_hooks("app_title", app_name=app) or [app])[0]


def _exception_kind(name: str) -> str:
	"""What Frappe says about an exception class it defines: the kind of error it is and the HTTP
	status it answers with. Empty for any other exception."""
	cls = getattr(frappe.exceptions, name or "", None)
	if not (isinstance(cls, type) and issubclass(cls, Exception)):
		return ""
	parents = [b.__name__ for b in cls.__mro__[1:] if b.__module__ == frappe.exceptions.__name__]
	parts = [f"a kind of {parents[0]}"] if parents else []
	if getattr(cls, "http_status_code", None):
		parts.append(f"HTTP {cls.http_status_code}")
	return ", ".join(parts)


def _public_words(message: str, exception: str) -> str:
	"""Search words for a public lookup: the message up to where its filled-in values begin ("Could
	not find Customer: …" gives "Could not find Customer"), plus the exception's name in words.
	Empty when that leaves too little to search on."""
	fixed = VALUE_START.split(message, maxsplit=1)[0]
	named = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", exception or "")
	words = re.findall(r"[A-Za-z][A-Za-z\-]*", f"{fixed} {named}")
	return " ".join(words) if len(words) >= MIN_HELP_WORDS else ""


def _docs_help(message: str, exception: str) -> str | None:
	"""What the Frappe documentation, then a web search, say about a message nothing on this site
	explains; None when neither has anything or lookups are off."""
	words = _public_words(message, exception)
	if not words:
		return None
	from flow.tools.web import docs_howto

	from flow.tools.web import NOT_IN_DOCS

	try:
		answer = docs_howto.func(topic=words).get("answer")
	except Exception:
		return None
	# "Nothing in the docs" is no help with an error.
	return None if answer == NOT_IN_DOCS else answer


def _clean(error: str) -> tuple[str, str]:
	"""(the message itself, the exception class if the paste included a traceback)."""
	text = (error or "").strip()
	exception = ""
	matches = EXCEPTION_LINE.findall(text)
	if matches:
		exception, text = matches[-1]
	previous = None
	while previous != text:
		previous, text = text, CHROME_PREFIXES.sub("", text, count=1)
	lines = [line.strip() for line in text.splitlines() if line.strip()]
	return " ".join(lines)[:500], exception


def _can_read_source() -> bool:
	return frappe.session.user != "Guest" and frappe.get_cached_value(
		"User", frappe.session.user, "user_type"
	) == "System User"


def _find_in_source(message: str) -> list[dict[str, Any]]:
	"""Where the message is raised: the exact text first, then its longest fixed parts, then
	its leading words (a template's fixed start, "Could not find" of "Could not find {0}")."""
	for needle, translated_only in (
		[(n, False) for n in _needles(message)]
		+ [(n, False) for n in _trimmed_windows(message)]
		+ [(n, True) for n in _leading_words(message)]
	):
		exact = needle == message.strip(" .")
		hits = _search(needle, translated_only)
		if not exact:
			hits = [h for h in hits if _fits_message(h["file"], message)]
		if hits:
			hits.sort(key=lambda h: not any(m in h["line_text"] for m in RAISE_MARKERS))
			# A partial match is a guess: show only the best one.
			return hits[: MAX_SOURCE_MATCHES if exact else 1]
	return []


def _fits_message(file: str, message: str) -> bool:
	"""A partial match inside one DocType's own folder is relevant only when the message names that
	DocType: "Mandatory fields required in Sales Order" shares its start with a Payroll Entry
	script's message, which says nothing about where this one is raised."""
	found = re.search(r"/doctype/([^/]+)/", file.replace(os.sep, "/"))
	if not found:
		return True
	return found.group(1).replace("_", " ") in message.lower().replace("_", " ")


def _needles(message: str) -> list[str]:
	"""Search strings, most specific first. Values filled into a template ("Row #1", quoted
	names, numbers) split the message into the fixed parts the source code contains."""
	parts = re.split(r"\"[^\"]*\"|'[^']*'|#\d+|\d[\d,.:-]*|<[^>]+>|:\s|\[[^\]]*\]", message)
	# Multi-word phrases only: a lone identifier ("delivery_date") matches unrelated code.
	fragments = sorted(
		{p.strip(" .,;") for p in parts if len(p.strip(" .,;")) >= MIN_FRAGMENT and " " in p.strip()},
		key=len,
		reverse=True,
	)
	whole = message.strip(" .")
	return ([whole] if " " in whole else []) + [f for f in fragments if f != whole][:4]


def _trimmed_windows(message: str) -> list[str]:
	"""The message with a few words cut from its start and/or end, longest first, for text the
	user wrote around the real message. Only multi-word runs, so they stay specific."""
	words = message.split()
	windows = []
	for cut in range(1, 2 * MAX_TRIMMED_WORDS + 1):
		for head in range(max(0, cut - MAX_TRIMMED_WORDS), min(cut, MAX_TRIMMED_WORDS) + 1):
			part = words[head : len(words) - (cut - head)]
			phrase = " ".join(part).strip(" .,;:")
			if len(part) >= MIN_WINDOW_WORDS and len(phrase) >= MIN_FRAGMENT and phrase not in windows:
				windows.append(phrase)
	return windows


def _leading_words(message: str) -> list[str]:
	"""The message's first 5..3 words, for templates whose filled-in values follow a fixed start."""
	words = re.findall(r"[A-Za-z][\w']*", message)
	out = []
	for n in (5, 4, 3):
		if len(words) >= n:
			phrase = " ".join(words[:n])
			if len(phrase) >= MIN_FRAGMENT and phrase not in out:
				out.append(phrase)
	return out


def _search(needle: str, translated_only: bool = False) -> list[dict[str, Any]]:
	"""Files whose source contains `needle` (case-insensitive), one hit per file. With
	`translated_only`, only user-facing strings count: lines that wrap text in _( or __(."""
	lowered = needle.lower()
	hits = []
	for path in _source_files():
		try:
			with open(path, encoding="utf-8", errors="ignore") as f:
				text = f.read()
		except OSError:
			continue
		if lowered not in text.lower():
			continue
		lines = text.splitlines()
		for i, line in enumerate(lines):
			if len(line) > MAX_LINE or lowered not in line.lower():
				continue
			if any(marker in line for marker in DEFINITION_MARKERS) or COLUMN_SPEC.search(line):
				continue
			if translated_only and not re.search(r"__?\(\s*[\"']" + re.escape(needle[:6]), line, re.IGNORECASE):
				continue
			hits.append(_hit(path, lines, i))
			break
		if len(hits) >= 10:
			break
	return hits


def _hit(path: str, lines: list[str], index: int) -> dict[str, Any]:
	relative = os.path.relpath(path, frappe.get_app_path("frappe", "..", ".."))
	app = relative.split(os.sep, 1)[0]
	return {
		"app": frappe.get_hooks("app_title", app_name=app)[0] if frappe.get_hooks("app_title", app_name=app) else app,
		"file": relative,
		"line": index + 1,
		"line_text": lines[index],
	}


@lru_cache(maxsize=1)
def _source_files() -> tuple[str, ...]:
	"""Source files of the installed apps (cached for the worker's lifetime)."""
	files = []
	for app in frappe.get_installed_apps():
		root = frappe.get_app_path(app)
		for dirpath, dirnames, filenames in os.walk(root):
			dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith(".")]
			files += [
				os.path.join(dirpath, name)
				for name in filenames
				if name.endswith(SOURCE_EXTENSIONS)
				and os.path.join(dirpath, name) != THIS_FILE
				and not name.endswith((".min.js", ".bundle.js"))
				and not name.startswith("test_")
			]
	return tuple(files)


def _find_translated_source(message: str) -> list[dict[str, Any]]:
	"""Where the English original of a translated message is raised (via the site's Translation
	records); empty when it isn't a translation or the user may not read source."""
	if not _can_read_source() or not frappe.has_permission("Translation", "read"):
		return []
	for original in frappe.get_list(
		"Translation", filters={"translated_text": message.strip(" .")}, pluck="source_text", limit=2
	):
		hits = _find_in_source(original)
		if hits:
			return hits
	return []


def _search_needles(message: str) -> list[str]:
	"""Short, specific strings to look for in the database: the message's fixed parts."""
	needles = [n for n in _needles(message) + _trimmed_windows(message) if len(n) <= 140]
	return list(dict.fromkeys(needles))[:5]


def _find_in_site_code(message: str) -> list[dict[str, str]]:
	"""Server Scripts, Client Scripts, print formats and notifications on this site that contain
	the message: an error raised from there is in no app's code."""
	if not _can_read_source():
		return []
	needles = [n for n in _search_needles(message) if len(n) >= MIN_SITE_FRAGMENT or n == message.strip(" .")]
	found: list[dict[str, str]] = []
	for doctype, fields in SITE_CODE:
		if not frappe.db.exists("DocType", doctype) or not frappe.has_permission(doctype, "read"):
			continue
		meta = frappe.get_meta(doctype)
		fields = [f for f in fields if meta.has_field(f)]
		if not fields:
			continue
		for needle in needles:
			names = frappe.get_list(
				doctype,
				or_filters=[[f, "like", f"%{needle}%"] for f in fields],
				pluck="name",
				limit=3,
			)
			if names:
				found += [{"doctype": doctype, "name": n} for n in names]
				break
	return found[:5]


def _recent_failures(message: str) -> list[dict[str, Any]]:
	"""The latest matching entry of each other failure log (failed scheduled jobs, emails that
	couldn't be sent, failed integration and webhook calls) the user may read."""
	needles = _search_needles(message)
	rows = []
	for doctype, field, filters, label in FAILURE_LOGS:
		if not frappe.db.exists("DocType", doctype) or not frappe.has_permission(doctype, "read"):
			continue
		if not frappe.get_meta(doctype).has_field(field):
			continue
		for needle in needles:
			found = frappe.get_list(
				doctype,
				filters={**filters, field: ["like", f"%{needle}%"]},
				fields=["name", "creation", field],
				order_by="creation desc",
				limit=1,
			)
			if found:
				lines = [ln.strip() for ln in (found[0].get(field) or "").splitlines() if ln.strip()]
				rows.append({"doctype": doctype, "label": label, "name": found[0].name,
					"creation": found[0].creation, "summary": lines[-1][:200] if lines else ""})
				break
	return rows


def _recent_error_logs(message: str, exception: str) -> list[dict[str, Any]]:
	if not frappe.has_permission("Error Log", "read"):
		return []
	needle = next((n for n in _needles(message) if len(n) <= 140), message[:140])
	filters = [["error", "like", f"%{needle}%"]]
	logs = frappe.get_list(
		"Error Log",
		filters=filters,
		fields=["name", "method", "creation", "error", "reference_doctype", "reference_name"],
		order_by="creation desc",
		limit=MAX_ERROR_LOGS,
	)
	return logs


def _format(
	message: str,
	exception: str,
	sources: list[dict[str, Any]],
	logs: list[dict[str, Any]],
	causes: list[tuple[str, str]],
	site_code: list[dict[str, str]] | None = None,
	failures: list[dict[str, Any]] | None = None,
	frames: list[dict[str, Any]] | None = None,
	help_text: str | None = None,
) -> str:
	out = ["**Error diagnosis**", f"> {message}"]
	if exception:
		kind = _exception_kind(exception)
		out.append(f"Exception type: `{exception}`" + (f" ({kind})" if kind else ""))

	if causes:
		out.append("**Likely cause**\n" + "\n".join(f"- {cause}" for cause, _fix in causes))
		out.append("**How to fix**\n" + "\n".join(f"- {fix}" for _cause, fix in causes))

	if sources or site_code or frames:
		# Location only — users want the cause and the fix, not the code.
		places = [f"- {f['app']} — `{f['file']}`, line {f['line']}, in `{f['function']}` (from the traceback)" for f in frames or []]
		places += [f"- {hit['app']} — `{hit['file']}`, line {hit['line']}" for hit in sources]
		places += [
			f"- {hit['doctype']} — [{hit['name']}](/desk/{hit['doctype'].lower().replace(' ', '-')}/{quote(hit['name'])})"
			f" (a script or template saved on this site)"
			for hit in site_code or []
		]
		out.append("**Where it comes from**\n" + "\n".join(places))
	elif _can_read_source() and not causes:
		out.append(
			"**Where it comes from**\nThis exact message isn't in the installed apps' code or in this site's "
			"scripts, print formats or notifications — it may come from a translation or a custom app."
		)

	for failure in failures or []:
		slug = failure["doctype"].lower().replace(" ", "-")
		out.append(
			f"**Recent failure in {failure['label']}**\n"
			f"[{failure['name']}](/desk/{slug}/{quote(failure['name'])}), "
			f"{frappe.utils.format_datetime(failure['creation'])}"
			+ (f"\n> {failure['summary']}" if failure["summary"] else "")
		)

	if logs:
		latest = logs[0]
		ref = (
			f" on {latest.reference_doctype} {latest.reference_name}"
			if latest.get("reference_doctype") and latest.get("reference_name")
			else ""
		)
		lines = [line.strip() for line in (latest.error or "").strip().splitlines() if line.strip()]
		summary = lines[-1][:200] if lines else ""
		out.append(
			f"**Recent occurrences** ({len(logs)} found in Error Log)\n"
			f"Latest: [{latest.name}](/desk/error-log/{latest.name}), {frappe.utils.format_datetime(latest.creation)}"
			f"{ref}{', in ' + latest.method if latest.get('method') else ''}"
			+ (f"\n> {summary}" if summary else "")
		)

	if help_text:
		out.append("**From the documentation**\n" + help_text)
	elif not (causes or sources or logs or site_code or failures or frames):
		out.append(
			"I couldn't find this message in the code or the Error Log, and it doesn't match a known "
			"error type. Tell me what you were doing when it appeared (which page, which button)."
		)
	return "\n\n".join(out)
