# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

from unittest.mock import MagicMock, patch

import frappe
from frappe.tests import IntegrationTestCase

PAGE = """
<div class="result results_links result--ad"><a class="result__a" href="https://ad.example">Ad</a></div>
<div class="result results_links"><a class="result__a" href="https://docs.example/page">Doc page</a>
<a class="result__snippet">How it works.</a></div>
<div class="result results_links"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fforum.example%2Ft%2F1">Forum</a>
<a class="result__snippet">A thread.</a></div>
"""


class TestWebSearch(IntegrationTestCase):
	def _search(self, query="how does x work"):
		from flow.tools.web import web_search

		response = MagicMock(text=PAGE)
		response.raise_for_status.return_value = None
		with patch("requests.post", return_value=response) as post:
			return web_search.func(query=query), post

	def test_results_skip_ads_and_unwrap_redirects(self):
		result, post = self._search()
		self.assertEqual([r["url"] for r in result["results"]], ["https://docs.example/page", "https://forum.example/t/1"])
		self.assertEqual(result["results"][0]["snippet"], "How it works.")
		self.assertEqual(post.call_args.kwargs["data"], {"q": "how does x work"})

	def test_can_be_turned_off(self):
		from flow.tools.web import web_search

		frappe.conf.flow_web_search_enabled = 0
		try:
			with self.assertRaises(ValueError):
				web_search.func(query="anything")
		finally:
			frappe.conf.flow_web_search_enabled = None

	def test_offered_to_the_browser_model_beside_the_chosen_tools(self):
		from flow.lib.agent import MAX_TOOLS, _relevant_tools

		schemas = [{"function": {"name": n}} for n in ("show_records", "count", "read", "find_doctypes", "create", "update", "describe", "web_search")]
		chosen = [t["function"]["name"] for t in _relevant_tools([{"role": "user", "content": "create a todo"}], schemas)]
		self.assertIn("web_search", chosen)
		self.assertLessEqual(len(chosen), MAX_TOOLS + 1)


class TestFrappeDocs(IntegrationTestCase):
	"""frappe_docs uses docs.frappe.io's own search, then the page's Markdown copy."""

	def _get(self, url, **kwargs):
		response = MagicMock()
		response.raise_for_status.return_value = None
		if "search.search" in url:
			response.json.return_value = {"message": {"results": [
				{"title": "Old <mark>Roles</mark>", "route": "erpnext/v12/user/roles", "content": "old"},
				{"title": "Cloud roles", "route": "cloud/role-permissions", "content": "cloud"},
				{"title": "<mark>Role</mark> Based Permissions", "route": "erpnext/role-based-permissions", "content": "current"},
			]}}
		else:
			response.text = '---\ntitle: "Role Based Permissions"\n---\n\nPermissions are set on **roles**.'
		return response

	def test_best_page_text_with_installed_app_and_current_pages_first(self):
		from flow.tools.web import frappe_docs

		installed = {"frappe", "framework", "erpnext"}
		with patch("requests.get", side_effect=self._get) as get, patch("flow.tools.web._installed_app_words", return_value=installed):
			result = frappe_docs.func(query="role permissions")
		self.assertEqual(result["page"]["url"], "https://docs.frappe.io/erpnext/role-based-permissions")
		self.assertEqual(result["page"]["title"], "Role Based Permissions")
		self.assertEqual(result["page"]["text"], "Permissions are set on **roles**.")
		self.assertEqual(get.call_args_list[-1].args[0], "https://docs.frappe.io/erpnext/role-based-permissions.md")
		# The old versioned page and other products' pages follow.
		self.assertEqual([m["url"].split(".io/")[1] for m in result["more"]], ["erpnext/v12/user/roles", "cloud/role-permissions"])


class TestDocsHowto(IntegrationTestCase):
	"""docs_howto answers a "how to add X to Y" question from one docs lookup, no chat model."""

	def _get(self, url, **kwargs):
		response = MagicMock()
		response.raise_for_status.return_value = None
		if "search.search" in url:
			response.json.return_value = {"message": {"results": [
				{"title": "Taxes", "route": "erpnext/taxes", "content": "taxes"},
				{"title": "Charges", "route": "erpnext/charges", "content": "charges"},
			]}}
		else:
			response.text = "---\ntitle: x\n---\n\nOpen the form.\n\n" + ("Add a row. " * 400)
		return response

	def _empty(self, url, **kwargs):
		response = MagicMock()
		response.raise_for_status.return_value = None
		response.json.return_value = {"message": {"results": []}}
		return response

	def test_add_to_questions_become_letters_only_search_words(self):
		from flow.tools.web import add_to_arguments

		self.assertEqual(
			add_to_arguments("How to add new tax component to the sales invoice?"),
			{"topic": "add new tax component sales invoice"},
		)
		# Record IDs, emails and numbers never reach the search.
		self.assertEqual(
			add_to_arguments("how do I add jane@example.com to project PRJ-0042 in the tracker"),
			{"topic": "add project in tracker"},
		)

	def test_other_questions_are_not_add_to_questions(self):
		from flow.tools.web import add_to_arguments

		for text in ("how does a purchase order work", "add", "guide me through life", "", None):
			self.assertIsNone(add_to_arguments(text), text)

	def test_answer_is_the_page_text_its_link_and_related_pages(self):
		from flow.tools.web import docs_howto

		with patch("requests.get", side_effect=self._get), patch("flow.tools.web._installed_app_words", return_value={"erpnext"}):
			result = docs_howto.func(topic="add tax component sales invoice")
		answer = result["answer"]
		self.assertIn("Open the form.", answer)
		self.assertIn("Source: https://docs.frappe.io/erpnext/taxes", answer)
		self.assertIn("(https://docs.frappe.io/erpnext/charges)", answer)
		self.assertLess(len(answer), 2600)

	def test_nothing_found_gives_no_answer_so_the_chat_model_takes_over(self):
		from flow.tools.web import docs_howto

		with patch("requests.get", side_effect=self._empty), patch("flow.tools.web._installed_app_words", return_value=set()):
			result = docs_howto.func(topic="add something nobody documented")
		self.assertNotIn("answer", result)
		self.assertIn("Nothing found", result["answer_hint"])

	def test_can_be_turned_off(self):
		from flow.tools.web import docs_howto

		frappe.conf.flow_web_search_enabled = 0
		self.addCleanup(lambda: frappe.conf.pop("flow_web_search_enabled", None))
		with self.assertRaises(ValueError):
			docs_howto.func(topic="add tax")
