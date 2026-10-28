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
		# Other products' pages follow; the old versioned page is left out.
		self.assertEqual([m["url"].split(".io/")[1] for m in result["more"]], ["cloud/role-permissions"])

	def test_old_versioned_pages_are_kept_only_when_nothing_current_exists(self):
		from flow.tools.web import _current_only

		old = {"title": "Old", "url": "https://docs.frappe.io/erpnext/v13/user/roles", "snippet": ""}
		new = {"title": "New", "url": "https://docs.frappe.io/erpnext/roles", "snippet": ""}
		self.assertEqual(_current_only([old, new]), [new])
		self.assertEqual(_current_only([old]), [old])


class TestPreferredApps(IntegrationTestCase):
	"""A topic two apps cover is answered from the docs of the app whose records this site holds."""

	def _results(self, *routes):
		response = MagicMock()
		response.raise_for_status.return_value = None
		response.json.return_value = {"message": {"results": [{"title": r, "route": r, "content": ""} for r in routes]}}
		return response

	def test_pages_of_the_app_the_site_uses_come_first(self):
		from flow.tools.web import _docs_search

		with (
			patch("requests.get", return_value=self._results("erpnext/lead", "crm/leads")),
			patch("flow.tools.web._installed_app_words", return_value={"erpnext", "crm"}),
			patch("flow.tools.web._preferred_apps", return_value=[{"crm"}, {"erpnext"}]),
		):
			urls = [p["url"].split(".io/")[1] for p in _docs_search("convert lead to deal")]
		self.assertEqual(urls, ["crm/leads", "erpnext/lead"])

	def test_search_order_decides_when_the_site_has_no_preference(self):
		from flow.tools.web import _docs_search

		with (
			patch("requests.get", return_value=self._results("erpnext/lead", "crm/leads")),
			patch("flow.tools.web._installed_app_words", return_value={"erpnext", "crm"}),
			patch("flow.tools.web._preferred_apps", return_value=[]),
		):
			urls = [p["url"].split(".io/")[1] for p in _docs_search("convert lead to deal")]
		self.assertEqual(urls, ["erpnext/lead", "crm/leads"])

	def test_preferred_apps_come_from_the_doctypes_that_hold_records(self):
		from flow.tools.web import _preferred_apps

		doctypes = [frappe._dict(name="Alpha Thing", module="Mod A"), frappe._dict(name="Beta Thing", module="Mod B")]
		counts = {"Alpha Thing": 0, "Beta Thing": 7}
		with (
			patch("frappe.get_all", side_effect=lambda dt, **k: doctypes if dt == "DocType" else [("Mod A", "app_a"), ("Mod B", "app_b")]),
			patch("frappe.db.count", side_effect=lambda dt: counts[dt]),
			patch("frappe.get_hooks", return_value=[]),
		):
			self.assertEqual(_preferred_apps("create a thing"), [{"app_b"}])


class TestDocsHowto(IntegrationTestCase):
	"""docs_howto answers a how-to question from the part of a docs page that fits it, no chat model."""

	PAGES = {
		"app/intro": "---\ntitle: x\n---\n\nWelcome to the app.\n\n## Overview\n\nThings live here.",
		"app/widgets": (
			"A widget is a thing.\n\n## Making a widget\n\nOpen the form and save.\n\n"
			"## Converting a widget to a gadget\n\nPress **Convert**.\n\n![](/files/shot.png)\n\nThe widget becomes a gadget."
		),
	}

	def _get(self, url, **kwargs):
		response = MagicMock()
		response.raise_for_status.return_value = None
		if "search.search" in url:
			response.json.return_value = {"message": {"results": [
				{"title": "Intro", "route": "app/intro", "content": "intro"},
				{"title": "Widgets", "route": "app/widgets", "content": "widgets"},
			]}}
		else:
			response.text = self.PAGES[url.removeprefix("https://docs.frappe.io/").removesuffix(".md")]
		return response

	def _empty(self, url, **kwargs):
		response = MagicMock()
		response.raise_for_status.return_value = None
		response.json.return_value = {"message": {"results": []}}
		return response

	def test_topic_is_the_letters_only_words_of_the_message(self):
		from flow.tools.web import topic_arguments

		self.assertEqual(topic_arguments("How to convert a widget to a gadget?"), {"topic": "how to convert a widget to a gadget"})
		# IDs, emails and numbers never reach the search.
		self.assertEqual(topic_arguments("how do I add jane@example.com to PRJ-0042 in the tracker"), {"topic": "how do i add to in the tracker"})
		for text in ("", "???", "12 34", None):
			self.assertIsNone(topic_arguments(text), text)

	def test_the_section_that_fits_the_words_is_picked_over_the_page_opening(self):
		with patch("requests.get", side_effect=self._get), patch("flow.tools.web._installed_app_words", return_value={"app"}):
			from flow.tools.web import docs_howto

			result = docs_howto.func(topic="how to convert a widget to a gadget")
		answer = result["answer"]
		self.assertIn("(Widgets: Converting a widget to a gadget)", answer)
		self.assertIn("Press **Convert**.", answer)
		self.assertIn("Source: https://docs.frappe.io/app/widgets", answer)
		self.assertNotIn("Open the form and save.", answer)
		self.assertNotIn("shot.png", answer)  # image links don't load in a chat
		self.assertIn("(https://docs.frappe.io/app/intro)", answer)

	def test_a_page_deeper_in_the_results_wins_when_its_section_fits_better(self):
		from flow.tools.web import pick_section

		pages = [({"url": "a"}, "Intro to the app.\n\n## Overview\n\nSomething else."), ({"url": "b"}, "## Converting a widget to a gadget\n\nPress Convert.")]
		page, heading, body, score = pick_section(pages, "convert widget gadget")
		self.assertEqual((page["url"], heading, body), ("b", "Converting a widget to a gadget", "Press Convert."))
		self.assertGreater(score, 0)

	def test_words_found_in_every_part_do_not_decide(self):
		from flow.tools.web import pick_section

		pages = [({"url": "a"}, "the how to start.\n\n## One\n\nthe how to.\n\n## Two\n\nthe how to gadget.")]
		self.assertEqual(pick_section(pages, "how to the")[1::2], (None, 0.0))  # nothing distinguishes a part: the opening
		self.assertEqual(pick_section(pages, "how to gadget")[1], "Two")

	def test_when_the_docs_have_nothing_the_web_results_are_the_answer(self):
		from flow.tools.web import docs_howto

		results = [{"title": "A forum thread", "url": "https://forum.example/t/1", "snippet": "Try the form."}]
		with patch("requests.get", side_effect=self._empty), patch("flow.tools.web._installed_app_words", return_value=set()), patch(
			"flow.tools.web._duckduckgo", return_value=results
		) as web:
			result = docs_howto.func(topic="something nobody documented")
		web.assert_called_once_with("something nobody documented")
		self.assertIn("couldn't find this in the Frappe documentation", result["answer"])
		self.assertIn("- [A forum thread](https://forum.example/t/1) — Try the form.", result["answer"])

	def test_docs_that_mention_none_of_the_words_count_as_having_nothing(self):
		from flow.tools.web import docs_howto

		with patch("requests.get", side_effect=self._get), patch("flow.tools.web._installed_app_words", return_value={"app"}), patch(
			"flow.tools.web._duckduckgo", return_value=[{"title": "Elsewhere", "url": "https://web.example/x", "snippet": ""}]
		):
			result = docs_howto.func(topic="zebra xylophone")
		self.assertIn("web results may help", result["answer"])
		self.assertNotIn("Frappe documentation says", result["answer"])

	def test_nothing_in_the_docs_or_on_the_web_gives_no_answer_so_the_chat_model_takes_over(self):
		from flow.tools.web import docs_howto

		with patch("requests.get", side_effect=self._empty), patch("flow.tools.web._installed_app_words", return_value=set()), patch(
			"flow.tools.web._duckduckgo", return_value=[]
		):
			result = docs_howto.func(topic="something nobody documented")
		self.assertNotIn("answer", result)
		self.assertIn("Nothing found", result["answer_hint"])

	def test_a_web_search_that_errors_leaves_it_to_the_chat_model(self):
		from flow.tools.web import docs_howto

		with patch("requests.get", side_effect=self._empty), patch("flow.tools.web._installed_app_words", return_value=set()), patch(
			"flow.tools.web._duckduckgo", side_effect=OSError("offline")
		):
			self.assertNotIn("answer", docs_howto.func(topic="anything at all"))

	def test_can_be_turned_off(self):
		from flow.tools.web import docs_howto

		frappe.conf.flow_web_search_enabled = 0
		self.addCleanup(lambda: frappe.conf.pop("flow_web_search_enabled", None))
		with self.assertRaises(ValueError):
			docs_howto.func(topic="anything")
