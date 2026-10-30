# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""End-to-end coverage of the create path across many DocTypes.

For each DocType, a record is built the way a small model sends it — field labels as keys,
dates typed in the site's format, Link values by display name, a lone object instead of a
list — from values a user "said". It must pass every create precheck, save, and store what
was meant. A made-up value must be rejected, and creation_steps must produce a guide.
Each case runs in its own rolled-back transaction. DocTypes from apps that aren't installed
(or that lack the data a case needs) are skipped.
"""

import frappe
from frappe.tests import IntegrationTestCase
from frappe.utils import add_days, formatdate, nowdate

from flow.lib.agent import _grounding_text
from flow.tools.builtins import (
	KEY_ROW_FIELDS,
	_conditionally_required,
	_field_facts,
	_important_fields,
	_precheck_create,
	_user_filled,
	create,
	creation_steps,
)

# DocTypes whose minimal record saves. Journal Entry (debits must equal credits), Leave
# Application (needs a leave allocation) and Stock Entry (field use depends on the entry
# type) enforce business rules a generic record can't meet; their errors reach the model.
DOCTYPES = (
	"ToDo",
	"Note",
	"Event",
	"Contact",
	"Address",
	"Customer",
	"Supplier",
	"Item",
	"Lead",
	"Opportunity",
	"Quotation",
	"Sales Order",
	"Sales Invoice",
	"Delivery Note",
	"Purchase Order",
	"Purchase Invoice",
	"Purchase Receipt",
	"Material Request",
	"Payment Entry",
	"Employee",
	"Department",
	"Designation",
	"Attendance",
	"Project",
	"Task",
	"Timesheet",
	"Issue",
	"Warehouse",
	"Item Group",
	"Customer Group",
)


class TestCreateCoverage(IntegrationTestCase):
	def setUp(self):
		self.site_format = frappe.db.get_single_value("System Settings", "date_format") or "yyyy-mm-dd"

	def tearDown(self):
		frappe.db.rollback()

	def test_every_doctype(self):
		for doctype in DOCTYPES:
			with self.subTest(doctype=doctype):
				if not frappe.db.exists("DocType", doctype):
					continue
				_ensure_supplier()  # inside each case's transaction: every case is rolled back
				self._check(doctype)
				frappe.db.rollback()

	def _check(self, doctype):
		answer = creation_steps(doctype=doctype)["answer"]
		self.assertTrue(answer.startswith("1. "))
		self.assertIn("**Save**", answer)

		built = self._build(doctype)
		if built is None:
			self.skipTest(f"no data on this site for a required field of {doctype}")
		said, record, expected = built
		context = _grounding_text([{"role": "user", "content": f"create a {doctype}: {said}"}])

		args = {"doctype": doctype, "records": record}
		self.assertIsNone(_precheck_create(args, context))

		invented = dict(args["records"][0])
		key = next((k for k, v in invented.items() if isinstance(v, str) and k != "company"), None)
		if key:
			invented[key] = "Zzinvented Value"
			self.assertIsNotNone(_precheck_create({"doctype": doctype, "records": [invented]}, context))

		result = create(doctype=doctype, records=args["records"])
		self.assertEqual(result.get("failures"), None, result.get("failures"))
		doc = frappe.get_doc(doctype, result["created"][0])
		for fieldname, value in expected.items():
			self.assertTrue(_same(doc.get(fieldname), value), f"{doctype}.{fieldname}: {doc.get(fieldname)!r}")

	def _build(self, doctype):
		"""(what the user said, the record a model sends, fieldname -> value to be stored)."""
		meta = frappe.get_meta(doctype)
		_defaults, sole_links = _field_facts(doctype)
		required = _user_filled(meta)
		conditional = _conditionally_required(meta)
		extra = _important_fields(meta, exclude={f.fieldname for f in required + conditional})
		said, record, expected = [], {}, {}
		for field in required + conditional[:1] + extra:
			if field.fieldname in sole_links:
				continue
			value = self._value(field, expected)
			if value is None:
				if field in required:
					return None
				continue
			typed, stored = value
			said.append(f"{field.label}: {typed}")
			record[field.label] = typed
			expected[field.fieldname] = stored
		for table in meta.get_table_fields():
			if not table.reqd:
				continue
			child = frappe.get_meta(table.options)
			fields = [child.get_field(f) for f in KEY_ROW_FIELDS if child.get_field(f)]
			fields += [f for f in _user_filled(child) if f.fieldname not in KEY_ROW_FIELDS]
			fields += _important_fields(child, exclude={f.fieldname for f in fields}, limit=2)
			row = {}
			for field in fields:
				value = self._value(field, {})
				if value is None:
					if field.reqd:
						return None
					continue
				said.append(f"{field.label}: {value[0]}")
				row[field.label] = value[0]
			record[table.label] = [row]
		return " ; ".join(said), record, expected

	def _value(self, field, siblings):
		"""(what the user types, what should be stored), or None if the site has no data."""
		if field.fieldtype == "Date":
			date = add_days(nowdate(), -9000 if "birth" in field.fieldname else 3)
			return formatdate(date, self.site_format), date
		if field.fieldtype == "Datetime":
			value = f"{add_days(nowdate(), 3)} 10:00:00"
			return value, value
		if field.fieldtype == "Select":
			options = [o for o in (field.options or "").split("\n") if o.strip()]
			return (options[0], options[0]) if options else None
		if field.fieldtype in ("Link", "Dynamic Link"):
			target = field.options if field.fieldtype == "Link" else siblings.get(field.options)
			if target == "DocType":
				return "Customer", "Customer"
			if not target:
				return None
			if field.fieldname == "item_code":
				names = frappe.get_all("Item", filters={"is_stock_item": 1}, pluck="name", limit=1)
			else:
				names = frappe.get_all(target, pluck="name", limit=1, order_by="creation asc")
			return (_display_name(target, names[0]), names[0]) if names else None
		if field.fieldtype == "Int":
			return "3", 3
		if field.fieldtype in ("Float", "Currency", "Percent"):
			return "100", 100
		if field.fieldtype == "Check":
			return None
		if field.options == "Phone":
			return "+919876543210", "+919876543210"
		if field.options == "Email":
			return "flowtest@example.com", "flowtest@example.com"
		value = f"Flowtest {field.label or field.fieldname}"
		return value, value


def _ensure_supplier():
	"""Purchase documents need a Supplier; a fresh site may have none."""
	if not frappe.db.exists("DocType", "Supplier") or frappe.db.count("Supplier"):
		return
	group = frappe.get_all("Supplier Group", pluck="name", limit=1)
	if group:
		frappe.get_doc(
			{"doctype": "Supplier", "supplier_name": "Flowtest Supplier", "supplier_group": group[0]}
		).insert()


def _display_name(doctype, name):
	meta = frappe.get_meta(doctype)
	for fieldname in (meta.title_field, f"{frappe.scrub(doctype)}_name"):
		if fieldname and meta.has_field(fieldname):
			title = frappe.db.get_value(doctype, name, fieldname)
			if isinstance(title, str) and title:
				return title
	return name


def _same(actual, expected):
	try:
		return float(actual) == float(expected)
	except (TypeError, ValueError):
		return str(actual) == str(expected)
