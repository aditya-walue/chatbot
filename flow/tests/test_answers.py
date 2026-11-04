# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

from unittest.mock import MagicMock

import frappe
from frappe.tests import IntegrationTestCase

from flow.lib.agent import Agent
from flow.tools.answers import (
	_route_document_flow,
	_route_explain_doctype,
	_route_show_records,
	document_flow,
	explain_doctype,
	role_permissions,
	show_records,
	where_to_find,
)


class TestAnswerRoutes(IntegrationTestCase):
	def tearDown(self):
		frappe.db.rollback()

	def test_doctype_word_is_not_guessed_as_a_record_id(self):
		from flow.tools.builtins import _doctype_names

		# A Role named like a DocType word ("Notification") must not turn a how-to question that
		# mentions the type into a lookup of that Role.
		word = next(
			w.title()
			for name in _doctype_names()
			for w in name.split()
			if len(w) >= 8 and w.isalpha() and not frappe.db.exists("Role", w.title())
		)
		frappe.get_doc({"doctype": "Role", "role_name": word}).insert()
		self.assertIsNone(_route_show_records(f"how can i assign a role to a new {word.lower()}?"))

	def test_record_by_id_and_lists_route(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "answers probe"}).insert()
		self.assertEqual(_route_show_records(f"show the details of todo {todo.name}"), {"doctype": "ToDo", "names": [todo.name]})
		self.assertEqual(_route_show_records("list all todos"), {"doctype": "ToDo"})
		self.assertEqual(_route_show_records("show open todos"), {"doctype": "ToDo", "filters": {"status": "Open"}})
		self.assertIsNone(_route_show_records("show broken todos"))
		self.assertIsNone(_route_show_records("list all widgets"))
		self.assertIsNone(_route_show_records("what is a todo?"))

	def test_show_records_formats_from_live_data(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "answers probe"}).insert()
		one = show_records(doctype="ToDo", names=[todo.name])["answer"]
		self.assertIn(f"[{todo.name}](/desk/todo/{todo.name})", one)
		listed = show_records(doctype="ToDo", filters={"status": "Open"})["answer"]
		self.assertIn("ToDo record", listed)

	def test_a_record_type_inside_a_longer_phrase_lists_it_and_says_what_was_not_used(self):
		self.assertEqual(
			_route_show_records("is there any todo that is added"), {"doctype": "ToDo", "unapplied": "that is added"}
		)
		# Too many words left over to be filler: left to the model.
		self.assertIsNone(_route_show_records("are there any todo items with a long list of other conditions"))
		listed = show_records(doctype="ToDo", unapplied="that is added")["answer"]
		self.assertTrue(listed.endswith('Not used as a filter: "that is added".'))
		self.assertNotIn("Not used as a filter", show_records(doctype="ToDo")["answer"])

	def test_a_concept_that_is_not_a_record_type_is_defined_from_the_docs(self):
		from unittest.mock import patch

		with patch("flow.tools.web.docs_howto") as docs:
			docs.func.return_value = {"answer": "Docs say what it is."}
			result = explain_doctype.func(doctype="Zzq Notarealtype")
		docs.func.assert_called_once_with(topic="what is Zzq Notarealtype")
		self.assertEqual(result["answer"], "Docs say what it is.")

	def test_where_to_find_points_to_the_list_and_new_form_from_frappes_own_links(self):
		from urllib.parse import urlparse

		from frappe.utils import get_url_to_form, get_url_to_list

		answer = where_to_find(doctype="ToDo")["answer"]
		self.assertIn("**ToDo** is in the **", answer)
		self.assertIn(f"[ToDo list]({urlparse(get_url_to_list('ToDo')).path})", answer)
		self.assertIn(f"[New ToDo]({urlparse(get_url_to_form('ToDo', 'new')).path})", answer)
		self.assertNotIn("search bar", answer)

	def test_where_to_find_has_no_route_of_its_own_and_the_model_is_offered_it(self):
		from flow.lib.agent import DEFAULT_TOOLS
		from flow.tools.builtins import BUILTIN_TOOLS

		tool = next(t for t in BUILTIN_TOOLS if t.name == "where_to_find")
		self.assertIsNone(tool.route)
		self.assertIn("where_to_find", DEFAULT_TOOLS)

	def test_a_definition_question_is_answered_before_the_record_list(self):
		from flow.tools.builtins import BUILTIN_TOOLS

		names = [t.name for t in BUILTIN_TOOLS]
		self.assertLess(names.index("explain_doctype"), names.index("show_records"))

	def test_definition_and_flow_routes(self):
		self.assertEqual(_route_explain_doctype("what is a todo?"), {"doctype": "ToDo"})
		self.assertIsNone(_route_explain_doctype("what is the meaning of life"))
		self.assertEqual(_route_document_flow("what happens after a note is submitted"), {"doctype": "Note"})
		self.assertIsNone(_route_document_flow("what happens after lunch"))
		self.assertIn("**ToDo** is a record type", explain_doctype(doctype="ToDo")["answer"])
		self.assertIn("**ToDo**", document_flow(doctype="ToDo")["answer"])

	def test_routed_answers_skip_the_model(self):
		model = MagicMock()
		result = Agent(model=model, tools=[show_records, explain_doctype, document_flow]).run("what is a todo?")
		model.chat.assert_not_called()
		self.assertIn("record type", result.output)

	def test_no_permission_raises(self):
		frappe.set_user("Guest")
		try:
			with self.assertRaises(PermissionError):
				show_records(doctype="User")
		finally:
			frappe.set_user("Administrator")

	def test_any_phrasing_and_case_insensitive_status(self):
		frappe.get_doc({"doctype": "ToDo", "description": "status case probe"}).insert()
		self.assertEqual(_route_show_records("any open todos?"), {"doctype": "ToDo", "filters": {"status": "Open"}})
		listed = show_records(doctype="ToDo", filters={"status": "open"})["answer"]
		self.assertNotIn("No ToDo records", listed)

	def test_record_id_with_any_verb_routes_but_writes_do_not(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "pull up probe"}).insert()
		self.assertEqual(_route_show_records(f"can you pull up todo {todo.name} for me"), {"doctype": "ToDo", "names": [todo.name]})
		self.assertIsNone(_route_show_records(f"close todo {todo.name}"))

	def test_invalid_status_is_reported_with_options(self):
		with self.assertRaisesRegex(ValueError, "Options:"):
			show_records(doctype="ToDo", filters={"status": "Pending Forever"})

	def test_date_words_become_dates(self):
		from flow.tools.answers import _tidy_filters

		tidy = _tidy_filters("ToDo", {"date": ["<", "today()"]})
		self.assertEqual(tidy["date"], ["<", frappe.utils.today()])

	def test_show_records_pagination(self):
		for i in range(3):
			frappe.get_doc({"doctype": "ToDo", "description": f"show pagination probe {i}"}).insert()
		res = show_records(
			doctype="ToDo",
			filters={"description": ["like", "%show pagination probe%"]},
			limit=2,
			start=1,
			order_by="creation asc",
		)
		self.assertIn("Showing records 2 to 3 of 3.", res["answer"])
		self.assertEqual(len(res["names"]), 2)

	def test_role_permissions_for_doctype(self):
		res = role_permissions(role="System Manager", doctype="ToDo")
		self.assertIn("System Manager", res["answer"])
		self.assertIn("ToDo", res["answer"])



class TestAssignmentSteps(IntegrationTestCase):
	"""assignment_steps reads where one DocType is attached to another from the metadata."""

	def _answer(self, text):
		from flow.tools.answers import _route_assignment_steps, assignment_steps

		args = _route_assignment_steps(text)
		return args, (assignment_steps.func(**args)["answer"] if args else None)

	def test_roles_go_on_the_users_table(self):
		args, answer = self._answer("how do I add a role to a user")
		self.assertEqual(args, {"item": "Role", "target": "User"})
		self.assertIn("**Roles Assigned** table", answer)

	def test_a_field_on_the_item_pointing_back(self):
		args, answer = self._answer("how to assign a todo to a role?")
		self.assertEqual(args, {"item": "ToDo", "target": "Role"})
		self.assertIn("Open the ToDo", answer)

	def test_dynamic_link_table(self):
		args, answer = self._answer("how can I link a contact to a user")
		self.assertIsNotNone(args)
		self.assertIn("Open the Contact", answer)

	def test_unrelated_or_unknown_doctypes_fall_through(self):
		self.assertEqual(self._answer("how do I link my car to the moon")[0], None)
		self.assertEqual(self._answer("how do I assign a note to a note")[0], None)

	def test_an_item_reaches_a_target_through_a_record_it_links_to(self):
		from flow.tools.answers import _assignment_paths, assignment_steps

		if not (frappe.db.exists("DocType", "Employee") and frappe.db.exists("DocType", "Project")):
			self.skipTest("needs Employee and Project")
		paths = _assignment_paths("Employee", "Project")
		self.assertTrue(paths)
		answer = assignment_steps.func(item="Employee", target="Project")["answer"]
		self.assertIn("connect through the **User** the Employee links to", answer)
		self.assertIn("table", answer)

	def test_a_loose_question_naming_two_record_types_is_answered_from_metadata_even_without_a_link(self):
		from flow.tools.answers import _assignment_paths, assignment_steps, loose_assignment

		# Two record types the metadata does not link: the strict route says nothing, the loose one
		# still reaches the tool, which then says there is no direct link.
		pair = next(
			(
				(a, b)
				for a, b in (("Note", "Role"), ("ToDo", "Note"), ("Note", "Language"), ("Role", "Language"))
				if not _assignment_paths(a, b)
			),
			None,
		)
		if not pair:
			self.skipTest("every candidate pair is linked on this site")
		text = f"is it possible to assign a {pair[0].lower()} to a {pair[1].lower()}?"
		args = loose_assignment(text)
		self.assertEqual(args, {"item": pair[0], "target": pair[1]})
		self.assertIn("no field or table", assignment_steps.func(**args)["answer"])
		self.assertIsNone(loose_assignment("is it possible to assign a thing to a place?"))


class TestLowStockAndLeave(IntegrationTestCase):
	def test_only_rows_at_or_below_their_level_are_low_and_the_thinnest_come_first(self):
		from flow.tools.answers import _below_reorder

		levels = [
			{"item": "A", "warehouse": "W1", "level": 10},
			{"item": "B", "warehouse": "W1", "level": 5},
			{"item": "C", "warehouse": "W2", "level": 8},
			{"item": "D", "warehouse": "W2", "level": 3},
		]
		quantities = {("A", "W1"): 4, ("B", "W1"): 5, ("D", "W2"): 9}  # C has no stock record at all
		low = _below_reorder(levels, quantities)
		self.assertEqual([r["item"] for r in low], ["C", "A", "B"])
		self.assertEqual(low[0]["have"], 0)

	def test_low_stock_answers_from_the_site_without_error(self):
		from flow.tools.answers import low_stock

		if not frappe.db.exists("DocType", "Bin"):
			self.skipTest("needs stock")
		answer = low_stock.func()["answer"]
		self.assertIsInstance(answer, str)
		self.assertTrue(answer)

	def test_on_leave_covers_the_day_and_rejects_a_bad_date(self):
		from flow.tools.answers import on_leave

		if not frappe.db.exists("DocType", "Leave Application"):
			self.skipTest("needs leave management")
		self.assertIn("on approved leave", on_leave.func()["answer"])
		with self.assertRaises(ValueError):
			on_leave.func(date="not a date")


class TestTotalsAndTrailingStatus(IntegrationTestCase):
	def test_a_status_after_the_record_type_filters_by_it(self):
		from flow.tools.answers import count_arguments, list_target

		self.assertEqual(list_target("todos in open"), {"doctype": "ToDo", "filters": {"status": "Open"}})
		self.assertEqual(list_target("todos that are closed"), {"doctype": "ToDo", "filters": {"status": "Closed"}})
		self.assertEqual(count_arguments("how many todos are in open"), {"doctype": "ToDo", "filters": {"status": "Open"}})
		# No status among the words, or a clause too long to be filler: left to the model.
		self.assertIsNone(list_target("todos in a hurry"))
		self.assertIsNone(list_target("todos that are open closed and cancelled all at the same time"))

	def test_the_amount_field_is_a_listed_currency_field_else_the_first_one(self):
		from flow.tools.answers import _amount_field

		self.assertIsNone(_amount_field("ToDo"))  # nothing to add up
		if frappe.db.exists("DocType", "Sales Invoice"):
			field = frappe.get_meta("Sales Invoice").get_field(_amount_field("Sales Invoice"))
			self.assertEqual(field.fieldtype, "Currency")

	def test_a_total_of_a_record_type_with_an_amount_is_a_sum_not_a_count(self):
		from flow.tools.answers import count_arguments, total_arguments

		self.assertEqual(total_arguments("what is the total of sales invoices?"), {"doctype": "Sales Invoice"})
		self.assertIsNone(count_arguments("what is the total of sales invoices?"))
		self.assertEqual(count_arguments("total number of sales invoices"), {"doctype": "Sales Invoice"})
		self.assertIsNone(total_arguments("how many sales invoices"))
		self.assertEqual(count_arguments("what is the total of todos"), {"doctype": "ToDo"})  # nothing to add up

	def test_a_record_type_ending_in_entries_is_not_cut_short(self):
		from flow.tools.builtins import _doctype_from_phrase

		self.assertEqual(_doctype_from_phrase("stock entries"), "Stock Entry")

	def test_total_adds_the_field_over_the_matching_records(self):
		from unittest.mock import patch

		from flow.tools.answers import total

		if not frappe.db.exists("DocType", "Sales Invoice"):
			self.skipTest("needs accounts")
		with patch("frappe.get_list", return_value=[{"total": 1234.5, "count": 3}]) as got:
			result = total.func(doctype="Sales Invoice", field="grand_total", filters={"status": "Paid"})
		self.assertEqual((result["total"], result["count"]), (1234.5, 3))
		self.assertIn("across 3 Sales Invoice records", result["answer"])
		self.assertIn("status = Paid", result["answer"])
		self.assertEqual(got.call_args.kwargs["fields"], [{"SUM": "grand_total", "as": "total"}, {"COUNT": "*", "as": "count"}])

	def test_total_names_the_numeric_fields_when_the_field_is_not_numeric(self):
		from flow.lib.tool import ToolError
		from flow.tools.answers import total

		with self.assertRaises(ValueError):
			total.func(doctype="ToDo")  # no amount field at all
		with self.assertRaises(ToolError) as caught:
			total.func(doctype="ToDo", field="description")
		self.assertEqual(str(caught.exception), "'description' is not a numeric field of ToDo.")
		self.assertIn("describe()", caught.exception.advice)

	def test_total_is_offered_to_the_model(self):
		from flow.lib.agent import DEFAULT_TOOLS
		from flow.tools.builtins import BUILTIN_TOOLS

		self.assertIn("total", [t.name for t in BUILTIN_TOOLS])
		self.assertIn("total", DEFAULT_TOOLS)


class TestStatusSteps(IntegrationTestCase):
	"""status_steps explains how a record reaches a status, from its metadata."""

	def _answer(self, text):
		from flow.tools.answers import _route_status_steps, status_steps

		args = _route_status_steps(text)
		return args, (status_steps.func(**args)["answer"] if args else None)

	def test_status_set_by_hand(self):
		args, answer = self._answer("how do I mark a todo as closed?")
		self.assertEqual(args, {"doctype": "ToDo", "status": "Closed"})
		self.assertIn("Closed", answer)

	def test_status_without_as(self):
		self.assertEqual(self._answer("how do I mark this todo closed")[0], {"doctype": "ToDo", "status": "Closed"})
		self.assertEqual(self._answer("how to set a todo to closed")[0], {"doctype": "ToDo", "status": "Closed"})

	def test_unknown_status_or_doctype_falls_through(self):
		self.assertIsNone(self._answer("how do I mark a todo as purple")[0])
		self.assertIsNone(self._answer("how do I mark my car as paid")[0])

	def test_paid_comes_from_the_connected_payment_document(self):
		if not frappe.db.exists("DocType", "Sales Invoice"):
			self.skipTest("ERPNext not installed")
		args, answer = self._answer("how can I make a sales invoice as paid?")
		self.assertEqual(args, {"doctype": "Sales Invoice", "status": "Paid"})
		self.assertIn("Payment Entry", answer)
		self.assertIn("submitted", answer)


class TestListPhrasing(IntegrationTestCase):
	"""List and count questions name records by a plain noun and a status word."""

	def test_more_list_phrasings(self):
		from flow.tools.answers import _route_show_records

		for text in ("is there any open todos?", "are there open todos", "do we have open todos", "do i have any open todos"):
			self.assertEqual(_route_show_records(text), {"doctype": "ToDo", "filters": {"status": "Open"}}, text)

	def test_count_with_status_word(self):
		from flow.tools.builtins import _route_count

		self.assertEqual(_route_count("how many open todos are there?"), {"doctype": "ToDo", "filters": {"status": "Open"}})

	def test_unknown_word_falls_through(self):
		from flow.tools.answers import _route_show_records

		self.assertIsNone(_route_show_records("show me broken todos"))

	def test_invented_filter_key_uses_the_select_field_with_that_value(self):
		from flow.tools.builtins import _precheck_query

		args = {"doctype": "ToDo", "filters": {"todo_state": "Open"}}
		self.assertIsNone(_precheck_query(args))
		self.assertEqual(args["filters"], {"status": "Open"})
		bad = {"doctype": "ToDo", "filters": {"todo_state": "purple"}}
		self.assertIn("one of: Open", _precheck_query(bad).advice)

	def test_plain_noun_picks_the_record_type(self):
		if not frappe.db.exists("DocType", "Sales Invoice"):
			self.skipTest("ERPNext not installed")
		from flow.tools.answers import list_target

		self.assertEqual(list_target("unpaid customer invoices"), {"doctype": "Sales Invoice", "filters": {"status": "Unpaid"}})
		self.assertEqual(list_target("unpaid supplier invoices"), {"doctype": "Purchase Invoice", "filters": {"status": "Unpaid"}})


class TestAskedFields(IntegrationTestCase):
	"""A question about one record shows the fields its words name, from the record's labels."""

	def test_question_words_pick_fields_by_label(self):
		from flow.tools.answers import asked_fields

		self.assertIn("description", asked_fields("ToDo", "what is the description of todo abc123"))
		self.assertEqual(asked_fields("ToDo", "show todo abc123", exclude="abc123"), [])

	def test_whole_label_match_ranks_first(self):
		from flow.tools.answers import asked_fields

		# "priority" covers the label "Priority" fully; it beats longer labels that mention it.
		self.assertEqual(asked_fields("ToDo", "what is the priority of this?")[0], "priority")

	def test_empty_best_match_gives_way_to_a_filled_one(self):
		from flow.tools.answers import asked_fields

		todo = frappe.get_doc({"doctype": "ToDo", "description": "tier probe"}).insert()
		fields = asked_fields("ToDo", "what is the description here?", exclude=todo.name, record=todo.name)
		self.assertIn("description", fields)

	def test_record_answer_includes_asked_field(self):
		from flow.tools.answers import _route_show_records, show_records

		todo = frappe.get_doc({"doctype": "ToDo", "description": "asked field probe", "reference_type": "User"}).insert()
		args = _route_show_records(f"todo {todo.name}: what is its reference type?")
		self.assertIn("reference_type", args.get("about", []))
		self.assertIn("Reference Type", show_records.func(**args)["answer"])

	def test_asked_fields_are_the_whole_answer(self):
		from flow.tools.answers import _route_show_records, show_records

		todo = frappe.get_doc(
			{"doctype": "ToDo", "description": "focus probe", "reference_type": "User", "priority": "High"}
		).insert()
		args = _route_show_records(f"todo {todo.name}: what is its reference type?")
		answer = show_records.func(**args)["answer"]
		self.assertIn("Reference Type", answer)
		# The key-field summary (priority, status…) is left out when the question named a field.
		self.assertNotIn("Priority", answer)
		self.assertNotIn("Status", answer)

	def test_unnamed_fields_still_show_the_summary(self):
		from flow.tools.answers import show_records

		todo = frappe.get_doc({"doctype": "ToDo", "description": "summary probe", "priority": "High"}).insert()
		answer = show_records.func(doctype="ToDo", names=[todo.name])["answer"]
		self.assertIn("Priority", answer)


class TestCountQuestions(IntegrationTestCase):
	"""Count questions: totals, breakdowns by a field, and an app or module scope."""

	def test_breakdown_by_a_field(self):
		from flow.tools.answers import count_arguments

		self.assertEqual(count_arguments("number of open todos by priority"), {"doctype": "ToDo", "filters": {"status": "Open"}, "group_by": "priority"})
		self.assertEqual(count_arguments("how many todos are there?"), {"doctype": "ToDo"})
		# A breakdown word that names no field is left to the chat model.
		self.assertIsNone(count_arguments("how many todos per flavour"))

	def test_count_with_an_existence_phrase_then_a_site_phrase(self):
		from flow.tools.answers import count_arguments

		for text in (
			"how many todos are there in system?",
			"how many todos are there in the system",
			"how many todos do we have on the site",
			"how many todos exist in the system",
		):
			self.assertEqual(count_arguments(text), {"doctype": "ToDo"}, text)
		# "in the system" still isn't a module scope.
		self.assertEqual(count_arguments("how many todos in the system"), {"doctype": "ToDo"})

	def test_grouped_count_answer(self):
		from flow.tools.builtins import count

		frappe.get_doc({"doctype": "ToDo", "description": "group probe", "priority": "High"}).insert()
		result = count.func(doctype="ToDo", group_by="priority")
		self.assertIn("High", result["groups"])
		self.assertIn("| Priority | Count |", result["answer"])

	def test_scope_by_app_or_module(self):
		from flow.tools.answers import _scope

		self.assertIn("core", _scope("frappe framework"))
		self.assertEqual(_scope("core"), {"core"})
		self.assertIsNone(_scope("the system"))


class TestSingleDocTypes(IntegrationTestCase):
	"""Singles (one settings record, no table) are explained and shown, never counted."""

	def test_single_is_explained_not_counted(self):
		from flow.tools.answers import explain_doctype, show_records
		from flow.tools.builtins import count

		answer = explain_doctype.func(doctype="System Settings")["answer"]
		self.assertIn("single settings page", answer)
		self.assertIn("single settings page", show_records.func(doctype="System Settings")["answer"])
		with self.assertRaises(ValueError):
			count.func(doctype="System Settings")



class TestDocTypeAccess(IntegrationTestCase):
	"""Who can do what on a DocType, from the site's own permission rules and user roles."""

	def setUp(self):
		from flow.tools.answers import doctype_access

		self.tool = doctype_access
		self.role = frappe.get_doc({"doctype": "Role", "role_name": "Flow Access Probe Role"}).insert()
		# A customised rule replaces the DocType's standard ones, so the probe role is the only row.
		frappe.get_doc(
			{"doctype": "Custom DocPerm", "parent": "ToDo", "parenttype": "DocType", "parentfield": "permissions", "role": self.role.name, "permlevel": 0, "read": 1, "create": 1}
		).insert(ignore_permissions=True)
		self.user = frappe.get_doc(
			{"doctype": "User", "email": "flow.access.probe@example.com", "first_name": "Probe", "send_welcome_email": 0, "roles": [{"role": self.role.name}]}
		).insert()

	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_names_the_roles_with_the_right_and_the_users_who_hold_them(self):
		answer = self.tool.func(doctype="ToDo", right="create")["answer"]
		self.assertIn("Roles that can **create** **ToDo**", answer)
		self.assertIn(f"**{self.role.name}** — 1 user: Probe ({self.user.name})", answer)
		self.assertIn("Administrator can always do everything.", answer)

	def test_a_right_the_role_lacks_is_not_listed(self):
		result = self.tool.func(doctype="ToDo", right="delete")
		self.assertEqual(result["roles"], [])
		self.assertIn("No role can **delete**", result["answer"])

	def test_no_right_lists_every_right_each_role_has(self):
		answer = self.tool.func(doctype="ToDo")["answer"]
		self.assertIn(f"Roles that have access to **ToDo**", answer)
		self.assertIn(f"**{self.role.name}** (Read, Create", answer)  # plus whatever a new rule ticks by default

	def test_unknown_right_names_the_valid_ones(self):
		with self.assertRaises(ValueError) as caught:
			self.tool.func(doctype="ToDo", right="teleport")
		self.assertIn("create", str(caught.exception))

	def test_a_disabled_user_does_not_count(self):
		frappe.db.set_value("User", self.user.name, "enabled", 0)
		self.assertIn(f"**{self.role.name}** — no users", self.tool.func(doctype="ToDo", right="read")["answer"])

	def test_without_permission_to_list_users_only_roles_are_shown(self):
		from unittest.mock import patch

		real = frappe.has_permission
		with patch("frappe.has_permission", side_effect=lambda doctype=None, *a, **k: False if doctype == "User" else real(doctype, *a, **k)):
			answer = self.tool.func(doctype="ToDo", right="read")["answer"]
		self.assertIn(f"- **{self.role.name}**\n", answer + "\n")
		self.assertIn("can't list users", answer)

	def test_who_can_questions_become_arguments(self):
		from flow.tools.answers import access_arguments

		self.assertEqual(access_arguments("who has the access to create sales invoice?"), {"doctype": "Sales Invoice", "right": "create"})
		self.assertEqual(access_arguments("which users have access to item"), {"doctype": "Item"})
		# Not a question about who may do something, or no DocType named.
		self.assertIsNone(access_arguments("show all sales invoices"))
		self.assertIsNone(access_arguments("who has access to the moon"))
		self.assertIsNone(access_arguments("how can i create a sales invoice"))


class TestRoleQuestions(IntegrationTestCase):
	"""Questions about roles and users that role_permissions must not be forced to answer."""

	def setUp(self):
		self.role = frappe.get_doc({"doctype": "Role", "role_name": "Flow Role Probe"}).insert()
		self.user = frappe.get_doc(
			{"doctype": "User", "email": "flow.role.probe@example.com", "first_name": "Probe", "send_welcome_email": 0, "roles": [{"role": self.role.name}]}
		).insert()

	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_role_permissions_rejects_a_doctype_or_placeholder_as_the_role(self):
		for bad in ("ToDo", "****", ""):
			with self.assertRaises(ValueError) as caught:
				role_permissions(role=bad)
			self.assertIn("doctype_access", str(caught.exception))

	def test_role_permissions_rejects_any_unknown_role_and_suggests_close_names(self):
		for bad in ("can create", "me"):
			with self.assertRaises(ValueError):
				role_permissions(role=bad)
		with self.assertRaises(ValueError) as caught:
			role_permissions(role="Flow Role Prob")
		self.assertIn("Flow Role Probe", str(caught.exception))

	def test_role_users_lists_the_users_holding_the_role(self):
		from flow.tools.answers import role_users

		answer = role_users.func(role=self.role.name)["answer"]
		self.assertIn(self.user.name, answer)
		with self.assertRaises(ValueError):
			role_users.func(role="No Such Role Anywhere")

	def test_role_users_route_needs_a_real_role_and_leaves_doctype_access_alone(self):
		from flow.tools.answers import role_users_arguments

		self.assertEqual(role_users_arguments(f"which users have the {self.role.name} role?"), {"role": self.role.name})
		self.assertIsNone(role_users_arguments("which users can create a ToDo"))
		self.assertIsNone(role_users_arguments("what roles does Administrator have"))
		self.assertIsNone(role_users_arguments("which users have the Nonexistent Probe role"))

	def test_user_roles_for_the_asker_and_for_another_user(self):
		from flow.tools.answers import user_roles, user_roles_arguments

		own = user_roles_arguments("what roles do I have?")
		self.assertEqual(own, {"user": None})
		self.assertTrue(own)  # an empty dict would read as "no route"
		self.assertEqual(user_roles_arguments(f"what roles does {self.user.name} have"), {"user": self.user.name})
		self.assertIsNone(user_roles_arguments("which roles can create a ToDo"))
		self.assertIn(self.role.name, user_roles.func(user=self.user.name)["answer"])
		frappe.set_user(self.user.name)
		self.assertIn("You have", user_roles.func()["answer"])
