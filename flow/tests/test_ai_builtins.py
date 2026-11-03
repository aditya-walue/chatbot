# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

import json

import frappe
from frappe.tests import IntegrationTestCase

from flow.tools.builtins import (
	BUILTIN_TOOLS,
	count,
	create,
	creation_steps,
	run_action,
	delete,
	describe,
	execute,
	find_doctypes,
	read,
	sync_builtin_tools,
	update,
)


class TestFindDoctypes(IntegrationTestCase):
	def test_search_matches_by_keyword(self):
		names = {r["name"] for r in find_doctypes(search="ToDo")}
		self.assertIn("ToDo", names)

	def test_filters_by_module(self):
		rows = find_doctypes(module="Core", limit=200)
		self.assertTrue(rows)
		self.assertTrue(all(r["module"] == "Core" for r in rows))

	def test_excludes_child_tables(self):
		names = {r["name"] for r in find_doctypes(search="Role", limit=200)}
		self.assertIn("Role", names)
		self.assertNotIn("Has Role", names)  # a child table that also matches "Role"

	def test_includes_single_doctypes(self):
		names = {r["name"] for r in find_doctypes(search="System Settings", limit=200)}
		self.assertIn("System Settings", names)  # a single DocType

	def test_respects_read_permission(self):
		frappe.set_user("Guest")
		try:
			names = {r["name"] for r in find_doctypes(search="User", limit=200)}
			self.assertNotIn("User", names)
		finally:
			frappe.set_user("Administrator")


class TestFindDoctypesWrongModule(IntegrationTestCase):
	def test_wrong_module_falls_back_to_name_search(self):
		names = {r["name"] for r in find_doctypes(search="ToDo", module="Accounts")}
		self.assertIn("ToDo", names)


class TestDescribe(IntegrationTestCase):
	def test_returns_fields_and_permissions(self):
		result = describe(doctype="ToDo")

		self.assertEqual(result["doctype"], "ToDo")
		fieldnames = {f["fieldname"] for f in result["fields"]}
		self.assertIn("description", fieldnames)
		self.assertEqual(set(result["permissions"]), {"read", "write", "create", "delete"})

	def test_excludes_layout_fields(self):
		result = describe(doctype="ToDo")
		types = {f["type"] for f in result["fields"]}
		self.assertNotIn("Section Break", types)
		self.assertNotIn("Column Break", types)

	def test_permission_denied_raises(self):
		frappe.set_user("Guest")
		try:
			with self.assertRaises(PermissionError):
				describe(doctype="User")
		finally:
			frappe.set_user("Administrator")


class TestRead(IntegrationTestCase):
	def tearDown(self):
		frappe.db.rollback()

	def test_reads_matching_records(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "ai builtin read probe"}).insert()

		rows = read(doctype="ToDo", filters={"description": "ai builtin read probe"})

		self.assertEqual([r["name"] for r in rows], [todo.name])

	def test_limit_is_capped(self):
		rows = read(doctype="DocType", limit=10_000)
		self.assertLessEqual(len(rows), 200)

	def test_returns_requested_fields(self):
		frappe.get_doc({"doctype": "ToDo", "description": "fields probe"}).insert()
		rows = read(doctype="ToDo", filters={"description": "fields probe"}, fields=["name", "description"])
		self.assertEqual(rows[0]["description"], "fields probe")

	def test_pagination_with_start(self):
		t1 = frappe.get_doc({"doctype": "ToDo", "description": "read pagination probe 1"}).insert()
		t2 = frappe.get_doc({"doctype": "ToDo", "description": "read pagination probe 2"}).insert()
		rows = read(
			doctype="ToDo",
			filters={"description": ["like", "%read pagination probe%"]},
			order_by="creation asc",
			limit=1,
			start=1,
		)
		self.assertEqual(len(rows), 1)
		self.assertEqual(rows[0]["name"], t2.name)


class TestCount(IntegrationTestCase):
	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_counts_matching_records(self):
		for _ in range(3):
			frappe.get_doc({"doctype": "ToDo", "description": "ai builtin count probe"}).insert()

		result = count(doctype="ToDo", filters={"description": "ai builtin count probe"})

		self.assertEqual(result["count"], 3)
		self.assertEqual(result["answer"], "There are **3** ToDo records.")
		# The number only: no record IDs, which show_records lists.
		self.assertNotIn("names", result)
		self.assertNotIn("/desk/", result["answer"])

	def test_one_and_zero_counts(self):
		frappe.get_doc({"doctype": "ToDo", "description": "ai builtin count single"}).insert()
		self.assertEqual(count(doctype="ToDo", filters={"description": "ai builtin count single"})["answer"], "There is **1** ToDo.")
		result = count(doctype="ToDo", filters={"description": "nonexistent descriptionxyz123"})
		self.assertEqual(result["count"], 0)
		self.assertIn("no **ToDo** records", result["answer"])

	def test_batch_display_names(self):
		from flow.tools.builtins import _batch_display_names

		t1 = frappe.get_doc({"doctype": "ToDo", "description": "batch display probe 1"}).insert()
		t2 = frappe.get_doc({"doctype": "ToDo", "description": "batch display probe 2"}).insert()
		titles = _batch_display_names("ToDo", [t1.name, t2.name])
		self.assertEqual(titles[t1.name], "batch display probe 1")
		self.assertEqual(titles[t2.name], "batch display probe 2")

	def test_permission_denied_raises(self):
		frappe.set_user("Guest")
		with self.assertRaises(frappe.PermissionError):
			count(doctype="User")


class TestDescribeRecordName(IntegrationTestCase):
	def test_reports_submittable_and_workflow(self):
		result = describe(doctype="ToDo")
		self.assertFalse(result["submittable"])
		self.assertIsNone(result["workflow"])

	def test_doctype_name_as_record_name_explains_the_fix(self):
		with self.assertRaisesRegex(ValueError, r'read\(doctype="ToDo", fields=\["name"\]\)'):
			describe(doctype="ToDo", name="ToDo")


class TestExecute(IntegrationTestCase):
	@classmethod
	def setUpClass(cls):
		cls.enterClassContext(cls.enable_safe_exec())
		super().setUpClass()

	def tearDown(self):
		frappe.db.rollback()

	def test_returns_result_variable(self):
		self.assertEqual(execute(code="result = 1 + 1", description="Add two numbers"), 2)

	def test_no_result_returns_none(self):
		self.assertIsNone(execute(code="x = 5", description="Set a variable"))

	def test_can_read_via_frappe(self):
		count = execute(code="result = frappe.db.count('DocType')", description="Count doctypes")
		self.assertIsInstance(count, int)

	def test_imports_are_blocked(self):
		with self.assertRaises(Exception):
			execute(code="import os\nresult = os.getcwd()", description="Get working directory")


class TestCreate(IntegrationTestCase):
	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_requires_confirmation(self):
		self.assertTrue(create.requires_confirmation)

	def test_creates_doc_with_values(self):
		result = create(doctype="ToDo", records=[{"description": "ai create probe"}])

		self.assertEqual(result["doctype"], "ToDo")
		self.assertEqual(len(result["created"]), 1)
		name = result["created"][0]
		self.assertTrue(frappe.db.exists("ToDo", name))
		self.assertEqual(frappe.db.get_value("ToDo", name, "description"), "ai create probe")

	def test_permission_denied_raises(self):
		frappe.set_user("Guest")
		with self.assertRaises(PermissionError):
			create(doctype="User", records=[{"email": "x@example.com"}])

	def test_missing_required_field_reported_as_failure(self):
		# ToDo.description is mandatory; the row fails validation and is reported, not raised.
		result = create(doctype="ToDo", records=[{}])

		self.assertEqual(result["created"], [])
		self.assertEqual(len(result["failures"]), 1)
		self.assertEqual(result["failures"][0]["row"], 0)

	def test_child_table_fields_at_top_level_are_explained(self):
		# `user` is a field of Note's `seen_by` child table (Note Seen By), not of Note itself.
		result = create(doctype="Note", records=[{"title": "misplaced child probe", "user": "Administrator"}])

		self.assertEqual(result["created"], [])
		error = result["failures"][0]["error"]
		self.assertIn("`seen_by`", error)
		self.assertIn('"seen_by": [{"user": ...}]', error)
		self.assertFalse(frappe.db.exists("Note", {"title": "misplaced child probe"}))


class TestUpdate(IntegrationTestCase):
	def setUp(self):
		self.todo = frappe.get_doc({"doctype": "ToDo", "description": "ai update probe"}).insert()

	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_requires_confirmation(self):
		self.assertTrue(update.requires_confirmation)

	def test_modifies_doc_fields(self):
		result = update(doctype="ToDo", names=[self.todo.name], values={"status": "Closed"})

		self.assertEqual(result["updated"], [self.todo.name])
		self.assertEqual(frappe.db.get_value("ToDo", self.todo.name, "status"), "Closed")

	def test_invalid_value_reported_as_failure(self):
		# Status is a Select field with a fixed set; an invalid value fails that row.
		result = update(doctype="ToDo", names=[self.todo.name], values={"status": "Not A Real Status"})

		self.assertEqual(result["updated"], [])
		self.assertEqual(len(result["failures"]), 1)

	def test_missing_record_reported_as_failure(self):
		result = update(doctype="ToDo", names=["does-not-exist"], values={"status": "Closed"})

		self.assertEqual(result["updated"], [])
		self.assertEqual(result["failures"][0]["name"], "does-not-exist")

	def test_permission_denied_reported_as_failure(self):
		frappe.set_user("Guest")
		result = update(doctype="ToDo", names=[self.todo.name], values={"status": "Closed"})

		self.assertEqual(result["updated"], [])
		self.assertEqual(len(result["failures"]), 1)


class TestDelete(IntegrationTestCase):
	def setUp(self):
		self.todo = frappe.get_doc({"doctype": "ToDo", "description": "ai delete probe"}).insert()

	def tearDown(self):
		frappe.set_user("Administrator")
		frappe.db.rollback()

	def test_requires_confirmation(self):
		self.assertTrue(delete.requires_confirmation)

	def test_removes_doc(self):
		result = delete(doctype="ToDo", names=[self.todo.name])

		self.assertEqual(result["deleted"], [self.todo.name])
		self.assertFalse(frappe.db.exists("ToDo", self.todo.name))

	def test_missing_record_reported_as_failure(self):
		result = delete(doctype="ToDo", names=["does-not-exist"])

		self.assertEqual(result["deleted"], [])
		self.assertEqual(result["failures"][0]["name"], "does-not-exist")

	def test_permission_denied_reported_as_failure(self):
		frappe.set_user("Guest")
		result = delete(doctype="ToDo", names=[self.todo.name])

		self.assertEqual(result["deleted"], [])
		self.assertEqual(len(result["failures"]), 1)


class TestSyncBuiltinTools(IntegrationTestCase):
	def tearDown(self):
		frappe.db.rollback()

	def test_creates_rows_for_all_builtins(self):
		sync_builtin_tools()
		for builtin in BUILTIN_TOOLS:
			self.assertTrue(frappe.db.exists("Flow Tool", builtin.name))

	def test_resolves_back_to_runtime_tool(self):
		sync_builtin_tools()
		doc = frappe.get_doc("Flow Tool", "describe")
		runtime = doc.to_tool()
		self.assertEqual(runtime.name, "describe")
		self.assertIn("doctype", runtime.parameters["properties"])

	def test_is_idempotent(self):
		sync_builtin_tools()
		sync_builtin_tools()
		count = frappe.db.count("Flow Tool", {"slug": "read"})
		self.assertEqual(count, 1)


class TestWritePrecheck(IntegrationTestCase):
	"""Unusable write arguments go back to the model instead of becoming an approval card."""

	def _invoke(self, tool, arguments):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[tool])
		return agent._invoke(ToolCall(id="c1", name=tool.name, arguments=arguments))

	def test_empty_create_is_rejected_before_confirmation(self):
		result = self._invoke(create, {"doctype": "ToDo", "records": [{}]})
		result = json.loads(result)
		self.assertIn("No field values", result["error"])
		self.assertIn("non-empty list", result["next"])

	def test_misplaced_child_fields_are_rejected_before_confirmation(self):
		result = self._invoke(create, {"doctype": "Note", "records": [{"title": "x", "user": "Administrator"}]})
		self.assertIn("`seen_by`", json.loads(result)["error"])

	def test_empty_names_are_rejected_before_confirmation(self):
		result = self._invoke(delete, {"doctype": "ToDo", "names": []})
		result = json.loads(result)
		self.assertIn("No record IDs", result["error"])
		self.assertIn("names must be", result["next"])

	def test_single_record_object_is_wrapped_in_a_list(self):
		from flow.lib.agent import Question

		arguments = {"doctype": "ToDo", "records": {"description": "x"}}
		result = self._invoke(create, arguments)
		self.assertIsInstance(result, Question)
		self.assertEqual(arguments["records"], [{"description": "x"}])

	def test_valid_create_still_asks_for_approval(self):
		from flow.lib.agent import Question

		result = self._invoke(create, {"doctype": "ToDo", "records": [{"description": "x"}]})
		self.assertIsInstance(result, Question)

	def test_approved_resume_wraps_single_record_object(self):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[create])
		call = ToolCall(id="c1", name="create", arguments={"doctype": "ToDo", "records": {"description": "resume wrap probe"}})
		result = json.loads(agent._resolve_confirmation(call, "Approve"))
		self.assertEqual(len(result["created"]), 1)
		frappe.db.rollback()


class TestCreationSteps(IntegrationTestCase):
	def test_answer_lists_route_required_fields_and_save(self):
		answer = creation_steps(doctype="ToDo")["answer"]
		self.assertIn("[New ToDo](/desk/todo/new)", answer)
		self.assertIn("- **Description**", answer)
		self.assertIn("**Save**", answer)

	def test_submittable_doctype_lists_items_and_submit(self):
		if not frappe.db.exists("DocType", "Sales Invoice"):
			self.skipTest("ERPNext not installed")
		answer = creation_steps(doctype="Sales Invoice")["answer"]
		self.assertIn("**Items** table", answer)
		self.assertIn("**Quantity**", answer)
		self.assertIn("**Submit**", answer)
		self.assertNotIn("Debit To", answer)  # defaults from the Company

	def test_permission_denied_raises(self):
		frappe.set_user("Guest")
		try:
			with self.assertRaises(PermissionError):
				creation_steps(doctype="User")
		finally:
			frappe.set_user("Administrator")

	def test_agent_shows_the_answer_without_another_model_call(self):
		from unittest.mock import MagicMock

		from flow.lib.agent import Agent
		from flow.lib.model import ChatResponse, ToolCall

		model = MagicMock()
		model.chat.return_value = ChatResponse(
			content=None, tool_calls=[ToolCall(id="c1", name="creation_steps", arguments={"doctype": "ToDo"})]
		)
		# Phrased so no route matches: the model picks the tool itself.
		result = Agent(model=model, tools=[creation_steps]).run("explain adding a todo")

		self.assertEqual(model.chat.call_count, 1)
		self.assertEqual(result.output, creation_steps(doctype="ToDo")["answer"])
		self.assertEqual(result.messages[-1], {"role": "assistant", "content": result.output})


class TestMadeUpValues(IntegrationTestCase):
	"""create/update values must come from the user or a tool result, not the model."""

	def _invoke(self, tool, arguments, said):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[tool])
		messages = [{"role": "user", "content": said}]
		return agent._invoke(ToolCall(id="c1", name=tool.name, arguments=arguments), messages)

	def test_invented_values_are_rejected(self):
		result = self._invoke(
			create,
			{"doctype": "ToDo", "records": [{"description": "John Smith onboarding", "date": "1985-04-23"}]},
			"create new todo",
		)
		error = json.loads(result)["error"]
		self.assertIn("description=John Smith onboarding", error)
		self.assertIn("date=1985-04-23", error)

	def test_values_the_user_gave_pass(self):
		from flow.lib.agent import Question

		result = self._invoke(
			create,
			{"doctype": "ToDo", "records": [{"description": "Call Vendor K.", "date": "2026-10-05", "priority": "Medium"}]},
			"create a todo: call vendor k on 5 Oct 2026",  # priority Medium is ToDo's default
		)
		self.assertIsInstance(result, Question)

	def test_update_values_are_checked(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "update grounding probe"}).insert()
		result = self._invoke(update, {"doctype": "ToDo", "names": [todo.name], "values": {"status": "Cancelled"}}, "close it")
		self.assertIn("status=Cancelled", json.loads(result)["error"])
		frappe.db.rollback()

	def test_date_with_only_the_right_year_is_rejected(self):
		result = self._invoke(
			create,
			{"doctype": "ToDo", "records": [{"description": "call vendor", "date": "2026-01-01"}]},
			"todo: call vendor on 01-09-2026",
		)
		self.assertIn("date=2026-01-01", json.loads(result)["error"])

	def test_date_typed_day_first_matches(self):
		from flow.lib.agent import Question

		result = self._invoke(
			create,
			{"doctype": "ToDo", "records": [{"description": "call vendor", "date": "2026-09-01"}]},
			"todo: call vendor on 01-09-2026",
		)
		self.assertIsInstance(result, Question)

	def test_options_listed_by_metadata_tools_do_not_count(self):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[create])
		messages = [
			{"role": "user", "content": "create todo call vendor"},
			{"role": "assistant", "content": None, "tool_calls": [
				{"id": "s1", "type": "function", "function": {"name": "creation_steps", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "s1", "content": "Priority (High / Medium / Low)"},
		]
		call = ToolCall(id="c1", name="create", arguments={"doctype": "ToDo", "records": [{"description": "call vendor", "priority": "High"}]})
		self.assertIn("priority=High", json.loads(agent._invoke(call, messages))["error"])


class TestRunActionPrecheck(IntegrationTestCase):
	def _invoke(self, arguments):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[run_action])
		return agent._invoke(ToolCall(id="c1", name="run_action", arguments=arguments), [])

	def test_missing_record_points_to_create(self):
		result = json.loads(self._invoke({"doctype": "ToDo", "names": ["Nobody Known"], "action": "submit"}))
		self.assertIn("No ToDo record with ID 'Nobody Known'", result["error"])
		self.assertIn("call create", result["next"])

	def test_unavailable_action_lists_the_valid_ones(self):
		todo = frappe.get_doc({"doctype": "ToDo", "description": "run action probe"}).insert()
		error = json.loads(self._invoke({"doctype": "ToDo", "names": [todo.name], "action": "submit"}))["error"]
		self.assertIn("'submit' is not an available action", error)
		frappe.db.rollback()


class TestMissingRequired(IntegrationTestCase):
	def test_missing_required_fields_are_named_before_approval(self):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[create])
		call = ToolCall(id="c1", name="create", arguments={"doctype": "ToDo", "records": [{"status": "Open"}]})
		error = json.loads(agent._invoke(call, [{"role": "user", "content": "add a todo"}]))["error"]
		self.assertIn("missing required fields: Description", error)


class TestNormalizeValues(IntegrationTestCase):
	def test_labels_become_fieldnames(self):
		from flow.tools.builtins import _normalize_values

		self.assertEqual(
			_normalize_values("ToDo", {"Description": "x", "Allocated To": "Administrator"}),
			{"description": "x", "allocated_to": "Administrator"},
		)

	def test_numbers_typed_as_text_become_numbers(self):
		from flow.tools.builtins import _normalize_values

		self.assertEqual(
			_normalize_values("ToDo", {"description": "x", "Description": "y"})["description"], "y"
		)
		if frappe.db.exists("DocType", "Item"):
			values = _normalize_values("Item", {"Opening Stock": "1,500", "Standard Selling Rate": "99.5"})
			self.assertEqual(values["opening_stock"], 1500.0)
			self.assertEqual(values["standard_rate"], 99.5)

	def test_site_format_dates_become_iso(self):
		from flow.tools.builtins import _to_iso_date

		site_format = frappe.db.get_single_value("System Settings", "date_format")
		frappe.db.set_single_value("System Settings", "date_format", "dd-mm-yyyy")
		try:
			self.assertEqual(_to_iso_date("01-06-2003"), "2003-06-01")
			self.assertEqual(_to_iso_date("01/06/2003"), "2003-06-01")
			self.assertEqual(_to_iso_date("2003-06-01"), "2003-06-01")
			self.assertEqual(_to_iso_date("next friday"), "next friday")
		finally:
			frappe.db.set_single_value("System Settings", "date_format", site_format)


class TestGeneralCreateChecks(IntegrationTestCase):
	def _invoke(self, arguments, said):
		from flow.lib.agent import Agent
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[create])
		return agent._invoke(ToolCall(id="c1", name="create", arguments=arguments), [{"role": "user", "content": said}])

	def test_value_inside_a_longer_typo_is_not_grounded(self):
		result = self._invoke(
			{"doctype": "ToDo", "records": [{"description": "call vendor", "date": "2002-06-01"}]},
			"todo: call vendor on 01-06-20023",
		)
		self.assertIn("date=2002-06-01", json.loads(result)["error"])

	def test_unknown_field_is_named_with_suggestion(self):
		result = self._invoke({"doctype": "ToDo", "records": [{"description": "x", "priorty": "High"}]}, "todo x priority high")
		error = json.loads(result)["error"]
		self.assertIn("'priorty' is not a field of ToDo", error)
		self.assertIn("did you mean priority", error)

	def test_link_by_display_name_resolves_to_id(self):
		from flow.lib.agent import Question

		user = frappe.get_doc("User", "Administrator")
		arguments = {"doctype": "ToDo", "records": [{"description": "x", "allocated_to": user.full_name}]}
		result = self._invoke(arguments, f"todo x for {user.full_name}")
		self.assertIsInstance(result, Question)
		self.assertEqual(arguments["records"][0]["allocated_to"], "Administrator")

	def test_unknown_link_value_is_reported(self):
		result = self._invoke(
			{"doctype": "ToDo", "records": [{"description": "x", "allocated_to": "Nobody Atall"}]},
			"todo x for nobody atall",
		)
		self.assertIn("No User with ID or name 'Nobody Atall'", json.loads(result)["error"])


class TestUpdateRedirect(IntegrationTestCase):
	def test_update_of_missing_records_becomes_a_create(self):
		from flow.lib.agent import Agent, Question
		from flow.lib.model import Model, ToolCall

		agent = Agent(model=Model(model_id="openai/gpt-4o-mini"), tools=[create, update])
		call = ToolCall(id="c1", name="update", arguments={"doctype": "ToDo", "names": ["UNKNOWN-0001"], "values": {"description": "call supplier"}})
		messages = [
			{"role": "user", "content": "todo: call supplier"},
			{"role": "assistant", "content": None, "tool_calls": [
				{"id": "c1", "type": "function", "function": {"name": "update", "arguments": "{}"}}
			]},
		]
		result = agent._invoke(call, messages)

		self.assertIsInstance(result, Question)
		self.assertEqual(call.name, "create")
		self.assertEqual(call.arguments["records"], [{"description": "call supplier"}])
		self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "create")

	def test_update_of_existing_record_is_left_alone(self):
		from flow.tools.builtins import _update_to_create

		todo = frappe.get_doc({"doctype": "ToDo", "description": "redirect probe"}).insert()
		self.assertIsNone(_update_to_create({"doctype": "ToDo", "names": [todo.name], "values": {"status": "Closed"}}))
		frappe.db.rollback()

	def test_unknown_key_with_email_value_suggests_email_field(self):
		from flow.tools.builtins import _unknown_fields_error

		error = _unknown_fields_error("User", [{"user_name": "someone@example.com", "first_name": "A"}])
		self.assertIn("did you mean email", error)

	def test_unknown_key_whose_value_fits_one_field_is_mapped(self):
		from flow.tools.builtins import _normalize_values

		values = _normalize_values("User", {"user_name": "someone@example.com", "first_name": "A"})
		self.assertEqual(values, {"email": "someone@example.com", "first_name": "A"})



class TestRetiredBuiltin(IntegrationTestCase):
	def test_a_builtin_row_the_code_no_longer_lists_is_retired(self):
		from types import SimpleNamespace

		from flow.tools.builtins import retired_builtin

		def row(name, path="flow.tools.builtins.x", system=1):
			return SimpleNamespace(name=name, import_path=path, is_system_generated=system)

		self.assertTrue(retired_builtin(row("old_builtin")))  # no longer in BUILTIN_TOOLS
		self.assertFalse(retired_builtin(row("count")))
		# Only builtin rows made by the sync: a site's own imported tool is never retired.
		self.assertFalse(retired_builtin(row("old_builtin", system=0)))
		self.assertFalse(retired_builtin(row("my_tool", path="my_app.tools.my_tool")))


class TestRouting(IntegrationTestCase):
	def test_routes_recognise_unmistakable_requests(self):
		from flow.tools.builtins import _route_count, _route_creation_steps
		from flow.tools.diagnosis import _route_error

		self.assertEqual(_route_count("how many ToDos are there?"), {"doctype": "ToDo"})
		self.assertIsNone(_route_count("how many people work here"))
		self.assertEqual(_route_creation_steps("how to add new todo? give me the steps"), {"doctype": "ToDo"})
		self.assertEqual(_route_creation_steps("steps to create an note"), {"doctype": "Note"})
		self.assertIsNone(_route_creation_steps("create a todo with description call vendor"))
		self.assertEqual(_route_error("Could not find Row #1: Item: X"), {"error": "Could not find Row #1: Item: X"})
		self.assertIsNone(_route_error("how to create new todo?"))

	def test_routed_request_skips_the_model(self):
		from unittest.mock import MagicMock

		from flow.lib.agent import Agent

		model = MagicMock()
		result = Agent(model=model, tools=[count, creation_steps]).run("how many ToDos are there?")
		model.chat.assert_not_called()
		self.assertIn("ToDo", result.output)

	def test_stuck_model_stops_after_two_malformed_steps(self):
		from unittest.mock import MagicMock

		from flow.lib.agent import STUCK_MESSAGE, Agent
		from flow.lib.model import ChatResponse, ToolCall

		model = MagicMock()
		model.chat.side_effect = lambda *a, **k: ChatResponse(
			content=None, tool_calls=[ToolCall(id=frappe.generate_hash(length=6), name="invalid_tool_call", arguments={}, error="bad")]
		)
		result = Agent(model=model, tools=[count]).run("do something odd")
		self.assertEqual(result.output, STUCK_MESSAGE)
		self.assertEqual(model.chat.call_count, 2)


class TestQueryCleanup(IntegrationTestCase):
	def test_options_operators_and_labels_are_tidied(self):
		from flow.tools.builtins import _precheck_query

		args = {"doctype": "ToDo", "filters": {"order_by": "creation desc", "Priority": ["eq", "High"], "description": ["contains", "call"]}, "order_by": None}
		self.assertIsNone(_precheck_query(args))
		self.assertEqual(args["order_by"], "creation desc")
		self.assertEqual(args["filters"], {"priority": ["=", "High"], "description": ["like", "%call%"]})

	def test_unknown_filter_field_is_named(self):
		from flow.tools.builtins import _precheck_query

		self.assertIn("'priorty'", _precheck_query({"doctype": "ToDo", "filters": {"priorty": "not a priority"}}))
		# A misspelt key with a value that is an option of exactly one Select field uses that field.
		args = {"doctype": "ToDo", "filters": {"priorty": "High"}}
		self.assertIsNone(_precheck_query(args))
		self.assertEqual(args["filters"], {"priority": "High"})

	def test_read_without_fields_returns_key_columns(self):
		frappe.get_doc({"doctype": "ToDo", "description": "key columns probe"}).insert()
		rows = read(doctype="ToDo", filters={"description": "key columns probe"})
		self.assertIn("status", rows[0])
		frappe.db.rollback()


class TestCreateRouting(IntegrationTestCase):
	def test_values_parse_into_fields_and_item_rows(self):
		from flow.tools.builtins import _route_create

		self.assertEqual(
			_route_create("create a todo with description call vendor, priority High"),
			{"doctype": "ToDo", "records": [{"description": "call vendor", "priority": "High"}]},
		)
		self.assertIsNone(_route_create("create a todo for UNKNOWN-0001"))  # unknown part: model decides

	def test_request_without_values_asks_for_them(self):
		from flow.lib.agent import Agent
		from flow.tools.builtins import required_values
		from unittest.mock import MagicMock

		model = MagicMock()
		result = Agent(model=model, tools=[create, required_values]).run("create one todo")
		model.chat.assert_not_called()
		self.assertIn("To create a new **ToDo**, I need:", result.output)
		self.assertIn("**Description**", result.output)

	def test_routed_create_still_asks_for_approval(self):
		from flow.lib.agent import Agent
		from unittest.mock import MagicMock

		result = Agent(model=MagicMock(), tools=[create]).run("create a todo with description route probe")
		self.assertTrue(result.paused)
		self.assertIn("route probe", result.questions[0].prompt)


class TestSmallTalk(IntegrationTestCase):
	def test_greetings_and_thanks_route_without_the_model(self):
		from unittest.mock import MagicMock

		from flow.lib.agent import Agent
		from flow.tools.builtins import _route_small_talk, small_talk

		for text in ("hii", "Hello!", "hey there", "good morning"):
			self.assertEqual(_route_small_talk(text), {"kind": "greeting"}, text)
		self.assertEqual(_route_small_talk("thank you so much"), {"kind": "thanks"})
		self.assertIsNone(_route_small_talk("hi, how many employees are there?"))

		model = MagicMock()
		result = Agent(model=model, tools=[small_talk, count]).run("hii")
		model.chat.assert_not_called()
		self.assertIn("How can I help you today?", result.output)


class TestTypoTolerantRoutes(IntegrationTestCase):
	def test_misspelt_doctypes_route_but_ordinary_words_do_not(self):
		from flow.tools.builtins import _route_count, _route_creation_steps

		self.assertEqual(_route_count("how many todso are there"), {"doctype": "ToDo"})
		self.assertEqual(_route_count("how many notifcations are there?"), {"doctype": "Notification"})
		self.assertEqual(_route_creation_steps("how to create a new notifcation"), {"doctype": "Notification"})
		self.assertIsNone(_route_count("how many people work here"))
		self.assertIsNone(_route_count("how many systems"))


class TestCreateByName(IntegrationTestCase):
	def test_named_requests_fill_the_title_field_and_are_not_how_tos(self):
		from flow.tools.builtins import _laya_creation_steps, _parse_create_request

		for text in ("create a todo named probe", "create a todo with name probe"):
			doctype, record = _parse_create_request(text) or (None, None)
			self.assertEqual(doctype, "ToDo")
			self.assertEqual(list(record.values()), ["probe"])
			# A command with details is a write for the approval card, however Laya classifies it.
			self.assertIsNone(_laya_creation_steps(text))
		self.assertEqual(_laya_creation_steps("how do I create a todo"), {"doctype": "ToDo"})


class TestPrechecksCatchWhatTheToolWouldReport(IntegrationTestCase):
	def test_create_names_an_unknown_doctype_first(self):
		from flow.tools.builtins import _precheck_create

		error = _precheck_create({"doctype": "No Such Doctype", "records": [{"title": "x"}]})
		self.assertIn("is not a record type on this site", error)

	def test_read_names_an_unknown_field(self):
		from flow.tools.builtins import _precheck_query

		error = _precheck_query({"doctype": "ToDo", "fields": ["status", "no_such_field"]})
		self.assertIn("'no_such_field'", error)
		self.assertNotIn("'status'", error)

	def test_read_accepts_the_parent_columns_of_a_child_table(self):
		from flow.tools.builtins import _precheck_query

		self.assertIsNone(_precheck_query({"doctype": "Has Role", "fields": ["parent", "parenttype", "role"]}))

	def test_read_leaves_real_fields_standard_columns_and_expressions_alone(self):
		from flow.tools.builtins import _precheck_query

		for fields in (["status", "name", "creation"], ["_assign", "_comments"], ["count(name) as total"], ["*"], None):
			self.assertIsNone(_precheck_query({"doctype": "ToDo", "fields": fields}), fields)


class TestDidYouMean(IntegrationTestCase):
	NAMES = ["item_code", "item_group", "to_rename", "warehouse", "actual_qty", "stock_uom"]

	def test_a_name_that_shares_a_word_is_suggested(self):
		from flow.tools.builtins import _similar_names

		self.assertEqual(_similar_names("item_name", self.NAMES), ["item_code"])
		self.assertEqual(_similar_names("Item Name", self.NAMES), ["item_code"])

	def test_a_near_miss_in_spelling_is_suggested(self):
		from flow.tools.builtins import _similar_names

		self.assertEqual(_similar_names("warehous", self.NAMES), ["warehouse"])
		self.assertEqual(_similar_names("actual_qtty", self.NAMES), ["actual_qty"])

	def test_a_loose_match_is_not_suggested(self):
		from flow.tools.builtins import _did_you_mean, _similar_names

		self.assertNotIn("to_rename", _similar_names("item_name", self.NAMES))
		self.assertEqual(_did_you_mean("reserved_qty_xyz", ["to_rename", "warehouse"]), "")
		self.assertEqual(_did_you_mean("item_name", self.NAMES), " (did you mean item_code?)")

	def test_the_read_error_names_only_useful_suggestions(self):
		from flow.tools.builtins import _precheck_query

		error = _precheck_query({"doctype": "ToDo", "fields": ["descriptionn", "zzqx"]})
		self.assertIn("'descriptionn' (did you mean description?)", error)
		self.assertIn("'zzqx'", error)
		self.assertNotIn("'zzqx' (did you mean", error)


class TestUnknownRecordType(IntegrationTestCase):
	def test_the_closest_real_record_types_are_named(self):
		from flow.tools.builtins import _unknown_doctype

		error = _unknown_doctype("Todo Item")
		self.assertIn("'Todo Item' is not a record type on this site.", error)
		self.assertIn("ToDo", error)
		self.assertIn("find_doctypes", error.advice)

	def test_a_real_or_missing_name_is_not_an_error(self):
		from flow.tools.builtins import _unknown_doctype

		for value in ("ToDo", "", None, 5):
			self.assertIsNone(_unknown_doctype(value), value)

	def test_count_read_and_show_records_all_say_it_before_running(self):
		from flow.tools.answers import show_records
		from flow.tools.builtins import _precheck_query, count, read

		self.assertIn("not a record type", _precheck_query({"doctype": "Zzq Balance", "filters": {"a": 1}}))
		for tool in (count, read):
			self.assertIsNotNone(tool.precheck)
		self.assertIn("not a record type", show_records.precheck({"doctype": "Zzq Balance"}))
		self.assertIsNone(show_records.precheck({"doctype": "ToDo"}))


class TestFindDoctypesRanking(IntegrationTestCase):
	ROWS = [
		{"name": "Quick Stock Balance", "module": "Stock", "issingle": 1},
		{"name": "Stock Settings", "module": "Stock", "issingle": 1},
		{"name": "Stock Ledger Entry", "module": "Stock", "issingle": 0},
		{"name": "Stock Entry", "module": "Stock", "issingle": 0},
		{"name": "Bulk Stocktake", "module": "Stock", "issingle": 0},
		{"name": "Stock", "module": "Stock", "issingle": 0},
	]

	def test_the_exact_name_then_whole_word_matches_then_the_rest(self):
		from flow.tools.builtins import _rank_doctypes

		names = [r["name"] for r in _rank_doctypes(self.ROWS, "stock")]
		self.assertEqual(names[0], "Stock")
		self.assertEqual(names[1:3], ["Stock Entry", "Stock Ledger Entry"])  # record types, shorter name first
		self.assertEqual(names[-1], "Bulk Stocktake")  # the word is only part of another word
		self.assertLess(names.index("Stock Settings"), names.index("Bulk Stocktake"))
		self.assertLess(names.index("Stock Entry"), names.index("Stock Settings"))

	def test_a_plural_keyword_finds_the_singular_name(self):
		from flow.tools.builtins import _rank_doctypes

		self.assertEqual(_rank_doctypes(self.ROWS, "stocks")[0]["name"], "Stock")

	def test_no_keyword_keeps_the_order_given(self):
		from flow.tools.builtins import _rank_doctypes

		self.assertEqual(_rank_doctypes(self.ROWS, None), self.ROWS)

	def test_the_tool_returns_the_best_match_first_and_no_extra_keys(self):
		from flow.tools.builtins import find_doctypes

		found = find_doctypes.func(search="note", limit=3)
		self.assertTrue(found)
		self.assertEqual(found[0]["name"], "Note")
		self.assertEqual(set(found[0]), {"name", "module"})


class TestProblemsReadAsPlainStatements(IntegrationTestCase):
	"""What a tool call is refused for is a statement a person can read; advice is for the model."""

	def test_a_missing_required_field_names_the_fields_and_keeps_the_instruction_apart(self):
		from flow.lib.tool import problem_payload
		from flow.tools.builtins import _precheck_create

		problem = _precheck_create({"doctype": "Item", "records": [{"item_name": "Zz Probe Item"}]})
		self.assertIn("missing required fields: Item Code", str(problem))
		for wording in ("call create", "Ask the user", "describe(", "find_doctypes"):
			self.assertNotIn(wording, str(problem))
		payload = problem_payload(problem)
		self.assertEqual(payload["error"], str(problem))
		self.assertIn("call create again", payload["next"])

	def test_the_shown_error_is_the_statement_only(self):
		from flow.lib.agent import _error_text
		from flow.lib.tool import problem_payload
		import json

		from flow.tools.builtins import _unknown_doctype

		problem = _unknown_doctype("Todo Item")
		self.assertNotIn("find_doctypes", _error_text(json.dumps(problem_payload(problem))))
		self.assertIn("find_doctypes", problem_payload(problem)["next"])

	def test_a_problem_without_advice_has_no_next(self):
		from flow.lib.tool import Problem, problem_payload

		self.assertEqual(problem_payload(Problem("Plain.")), {"error": "Plain."})
		self.assertEqual(problem_payload("A plain string"), {"error": "A plain string"})
