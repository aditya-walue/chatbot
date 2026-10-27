# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Tools that look outside the site, for questions its own tools can't answer:

- frappe_docs  the official documentation at docs.frappe.io (Frappe Framework, ERPNext, HR,
               CRM, Helpdesk, Cloud…), through that site's own search: the best page's text
- web_search   the public web, when the documentation has nothing
- docs_howto   one documentation lookup answered as-is, for "how to add X to Y" questions that
               Laya routes (no chat-model call)

Uses DuckDuckGo's HTML results page — no API key, but an unofficial endpoint that can rate-limit
or change without notice. Only the query is sent: the words the model writes, or for docs_howto
the letters-only words of the user's question (a word with a digit, @ or . inside is dropped); no record data. Site config
`flow_web_search_enabled` = 0 turns it off.
"""

from __future__ import annotations

import re
from typing import Annotated, Any
from urllib.parse import parse_qs, urlparse

import frappe

from flow.lib.tool import tool

SEARCH_URL = "https://html.duckduckgo.com/html/"
DOCS_HOST = "docs.frappe.io"
DOCS_SEARCH_URL = f"https://{DOCS_HOST}/api/method/wiki.frappe_wiki.doctype.wiki_document.search.search"
DOCS_MORE_LIMIT = 4
# The model reads the page in a small context window; the rest is a link away.
DOCS_TEXT_LIMIT = 3500
RESULTS_LIMIT = 5
TIMEOUT_SECONDS = 10
QUERY_LIMIT = 200


@tool
def web_search(
	query: Annotated[str, 'Search words, e.g. "ERPNext give a role permission on a doctype". No record IDs or personal data.'],
) -> dict[str, Any]:
	"""Search the public web, only when no site tool or frappe_docs can answer. Answer from
	the snippets in your own words and name the source links."""
	if _turned_off():
		raise ValueError("Web search is turned off on this site.")
	query = " ".join(str(query or "").split())[:QUERY_LIMIT]
	if not query:
		raise ValueError("Give the search words.")
	results = _duckduckgo(query)
	if not results:
		return {"query": query, "results": [], "answer_hint": "No web results found."}
	return {"query": query, "results": results, "source": "DuckDuckGo"}


def _turned_off() -> bool:
	return str(frappe.conf.get("flow_web_search_enabled")).strip().lower() in ("0", "false", "no")


def _duckduckgo(query: str) -> list[dict[str, str]]:
	import requests
	from bs4 import BeautifulSoup

	try:
		response = requests.post(
			SEARCH_URL,
			data={"q": query},
			headers={"User-Agent": "Mozilla/5.0 (compatible; FlowAssistant/1.0)"},
			timeout=TIMEOUT_SECONDS,
		)
		response.raise_for_status()
	except requests.RequestException as e:
		raise ValueError(f"Web search is unavailable right now ({type(e).__name__}).") from e

	soup = BeautifulSoup(response.text, "html.parser")
	results = []
	for item in soup.select("div.result"):
		if "result--ad" in (item.get("class") or []):
			continue
		link = item.select_one("a.result__a")
		if not link or not link.get("href"):
			continue
		snippet = item.select_one(".result__snippet")
		results.append(
			{
				"title": link.get_text(" ", strip=True),
				"url": _real_url(link["href"]),
				"snippet": snippet.get_text(" ", strip=True) if snippet else "",
			}
		)
		if len(results) >= RESULTS_LIMIT:
			break
	return results


def _real_url(href: str) -> str:
	"""DuckDuckGo sometimes wraps links in a redirect (//duckduckgo.com/l/?uddg=<url>)."""
	parsed = urlparse(href if "://" in href else f"https:{href}")
	if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
		target = parse_qs(parsed.query).get("uddg")
		if target:
			return target[0]
	return href


@tool
def frappe_docs(
	query: Annotated[str, 'What to look up, e.g. "role permissions manager" or "make payment entry against sales invoice".'],
) -> dict[str, Any]:
	"""Look up the official Frappe / ERPNext docs (docs.frappe.io): how a feature works, steps,
	what a setting means. Answer from the returned text in your own words and give its page link.
	Try the site's own tools first for anything about this site's data."""
	return _lookup_docs(query)


def _lookup_docs(query: str) -> dict[str, Any]:
	if _turned_off():
		raise ValueError("Looking things up online is turned off on this site.")
	query = " ".join(str(query or "").split())[:QUERY_LIMIT]
	if not query:
		raise ValueError("Give the words to look up.")
	pages = _docs_search(query)
	if not pages:
		return {"query": query, "page": None, "more": [], "answer_hint": "Nothing found in the Frappe documentation."}
	best = pages[0]
	return {
		"query": query,
		"page": {**best, "text": _page_text(best["url"]) or best["snippet"]},
		"more": [{"title": p["title"], "url": p["url"]} for p in pages[1:DOCS_MORE_LIMIT + 1]],
	}


ADD_VERBS = r"(?:add|attach|include|insert|put|apply)"
ADD_TO = re.compile(rf"\b{ADD_VERBS}\b.+\b(?:to|into|in|on)\b\s+\S", re.I)
QUESTION_WORDS = frozenset(
	"how to do does can could i you we should would the a an my our me please tell steps way".split()
)
ANSWER_LIMIT = 1800


def add_to_arguments(text: str) -> dict[str, Any] | None:
	"""The docs search words for "how to add X to Y" (any wording that adds something to
	something), or None. Only letters-only words are kept, so record IDs, emails and numbers in
	the message are never sent out."""
	if not isinstance(text, str) or not ADD_TO.search(text):
		return None
	tokens = (t.strip(".-") for t in re.split(r"[\s,;:!?()\"']+", text.lower()))
	words = [t for t in tokens if re.fullmatch(r"[a-z][a-z\-]*", t) and t not in QUESTION_WORDS]
	return {"topic": " ".join(words)} if words else None


@tool(final_answer=True)
def docs_howto(
	topic: Annotated[str, 'Search words for the docs, e.g. "add tax component sales invoice".'],
) -> dict[str, Any]:
	"""Answer a "how to add X to Y" question from one look at the Frappe / ERPNext docs
	(docs.frappe.io): the best page's text and its link. Use frappe_docs instead when you want
	to read the page yourself."""
	result = _lookup_docs(topic)
	page = result.get("page")
	if not page:
		return result  # no answer: the chat model takes over (and may try web_search)
	text = str(page.get("text") or page.get("snippet") or "")
	if len(text) > ANSWER_LIMIT:
		cut = text[:ANSWER_LIMIT]
		text = cut[: cut.rfind("\n\n")] if "\n\n" in cut else cut
	lines = [f"This is what the Frappe documentation says ({page.get('title') or 'page'}):", "", text.strip(), "", f"Source: {page['url']}"]
	if result.get("more"):
		lines += ["", "Related pages:"] + [f"- [{m['title']}]({m['url']})" for m in result["more"]]
	return {**result, "answer": "\n".join(lines)}


def _docs_search(query: str) -> list[dict[str, str]]:
	"""Pages docs.frappe.io's own search finds (it is a Frappe Wiki site), best first: pages of
	the apps this site runs before others (the docs also cover Frappe Cloud, Books…), and current
	pages before old versioned copies (…/v12/…)."""
	import requests

	try:
		response = requests.get(
			DOCS_SEARCH_URL,
			params={"query": query},
			headers={"User-Agent": "Mozilla/5.0 (compatible; FlowAssistant/1.0)"},
			timeout=TIMEOUT_SECONDS,
		)
		response.raise_for_status()
		found = (response.json().get("message") or {}).get("results") or []
	except (requests.RequestException, ValueError) as e:
		raise ValueError(f"The Frappe documentation can't be reached right now ({type(e).__name__}).") from e

	installed = _installed_app_words()
	pages = []
	for i, item in enumerate(found):
		route = str(item.get("route") or "").strip("/")
		if not route:
			continue
		parts = route.lower().split("/")
		pages.append(
			{
				"title": _plain(item.get("title")),
				"url": f"https://{DOCS_HOST}/{route}",
				"snippet": _plain(item.get("content"))[:300],
				"rank": (parts[0] not in installed, any(re.fullmatch(r"v\d+", p) for p in parts), i),
			}
		)
	pages.sort(key=lambda p: p.pop("rank"))
	return pages


def _plain(html: Any) -> str:
	return re.sub(r"<[^>]+>", "", str(html or "")).strip()


def _installed_app_words() -> set[str]:
	"""Names and title words of the installed apps ("erpnext", "hrms", "hr", "crm", "framework"…),
	to match docs sections (docs.frappe.io/<section>/...)."""
	words = set()
	for app in frappe.get_installed_apps():
		words.add(app.lower())
		title = (frappe.get_hooks("app_title", app_name=app) or [app])[0]
		words.update(w.lower() for w in title.split())
	return words


def _page_text(url: str) -> str:
	"""A docs page as Markdown (docs.frappe.io serves <page>.md), without its front matter."""
	import requests

	try:
		response = requests.get(
			f"{url}.md", headers={"User-Agent": "Mozilla/5.0 (compatible; FlowAssistant/1.0)"}, timeout=TIMEOUT_SECONDS
		)
		response.raise_for_status()
	except requests.RequestException:
		return ""
	text = response.text
	if text.startswith("---"):
		end = text.find("\n---", 3)
		text = text[end + 4 :] if end != -1 else text
	text = text.strip()
	return text if len(text) <= DOCS_TEXT_LIMIT else text[:DOCS_TEXT_LIMIT].rsplit("\n", 1)[0] + "\n…"

