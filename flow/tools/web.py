# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Tools that look outside the site, for questions its own tools can't answer:

- frappe_docs  the official documentation at docs.frappe.io (Frappe Framework, ERPNext, HR,
               CRM, Helpdesk, Cloud…), through that site's own search: the best page's text
- web_search   the public web, when the documentation has nothing
- docs_howto   one documentation lookup answered as-is (the matching part of the page), for
               how-to questions Laya routes that no site tool answers (no chat-model call)

Uses DuckDuckGo's HTML results page — no API key, but an unofficial endpoint that can rate-limit
or change without notice. Only the query is sent: the words the model writes, or for docs_howto
the letters-only words of the user's question (a word with a digit, @ or . inside is dropped); no record data. Site config
`flow_web_search_enabled` = 0 turns it off.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any
from urllib.parse import parse_qs, urlparse

import frappe

from flow.lib.tool import LayaRoute, tool

SEARCH_URL = "https://html.duckduckgo.com/html/"
DOCS_HOST = "docs.frappe.io"
DOCS_SEARCH_URL = f"https://{DOCS_HOST}/api/method/wiki.frappe_wiki.doctype.wiki_document.search.search"
DOCS_MORE_LIMIT = 4
DOCS_RERANK = 4  # pages docs_howto reads to find the section that fits best
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


ANSWER_LIMIT = 1800
SECTION_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.M)


def topic_arguments(text: str) -> dict[str, Any] | None:
	"""The docs search words for a how-to message: its letters-only words, so record IDs, emails
	and numbers in it are never sent out. None when there are none."""
	if not isinstance(text, str):
		return None
	tokens = (t.strip(".-") for t in re.split(r"[\s,;:!?()\"']+", text.lower()))
	words = [t for t in tokens if re.fullmatch(r"[a-z][a-z\-]*", t)]
	return {"topic": " ".join(words)} if words else None


def sections(text: str) -> list[tuple[str | None, str]]:
	"""A docs page's (heading, body) parts, the text before its first heading first. Image lines
	are dropped: their links are relative to the docs site and don't load in a chat."""
	heads = list(SECTION_HEADING.finditer(text))
	parts = [(None, text[: heads[0].start()] if heads else text)]
	for i, head in enumerate(heads):
		parts.append((head.group(1), text[head.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]))
	return [(heading, IMAGE_LINE.sub("", body).strip()) for heading, body in parts]


def pick_section(pages: list[tuple[Any, str]], topic: str) -> tuple[Any, str | None, str, float]:
	"""(page, heading, body, score) of the part, across `pages` of (page, text), that fits `topic`
	best. A topic word counts for more the fewer parts mention it ("how", "the" and "to" are in
	nearly all, so count for nothing), and three times as much in a heading as in the body. Ties
	keep search order; when nothing matches, the first page's opening with score 0."""
	words = {w[:5] for w in re.findall(r"[a-z]+", topic.lower())}
	parts = [(page, heading, body) for page, text in pages for heading, body in sections(text)]
	if not parts:
		return pages[0][0], None, "", 0.0
	haystacks = [f"{heading or ''}\n{body}".lower() for _page, heading, body in parts]
	weight = {w: math.log((len(parts) + 1) / (sum(w in h for h in haystacks) + 1)) for w in words}
	best, best_score = parts[0], 0.0
	for (page, heading, body), haystack in zip(parts, haystacks):
		score = sum(
			weight[w] * (3 if heading and w in heading.lower() else 1) for w in words if w in haystack
		)
		if score > best_score:
			best, best_score = (page, heading, body), score
	return (*best, best_score)


def _answering_section(topic: str) -> dict[str, Any] | None:
	"""The best (page, heading, text) for `topic` among the top few pages the docs search returns:
	the search ranks whole pages by loose matching, so the page whose own section fits the words
	best is picked instead of simply the first. None when the search finds nothing, or when no
	section mentions any of the words (the docs don't cover it)."""
	from concurrent.futures import ThreadPoolExecutor

	found = _docs_search(" ".join(str(topic or "").split())[:QUERY_LIMIT])[:DOCS_RERANK]
	if not found:
		return None
	with ThreadPoolExecutor(max_workers=len(found)) as pool:
		texts = list(pool.map(lambda p: _page_text(p["url"], None) or p["snippet"], found))
	page, heading, text, score = pick_section(list(zip(found, texts)), topic)
	if not score:
		return None
	return {"page": page, "heading": heading, "text": text, "more": [p for p in found if p is not page]}


@tool(final_answer=True, laya=LayaRoute("HOWTO", lambda text, signals: topic_arguments(text)))
def docs_howto(
	topic: Annotated[str, 'Search words for the docs, e.g. "convert lead to deal".'],
) -> dict[str, Any]:
	"""Answer a how-to question from the Frappe / ERPNext docs (docs.frappe.io): the part of the
	best page that matches, and its link. When the docs have nothing on it, the web search results
	instead. Use frappe_docs instead when you want to read the page yourself."""
	if _turned_off():
		raise ValueError("Looking things up online is turned off on this site.")
	found = _answering_section(topic)
	if not found:
		return _web_fallback(topic)
	page, heading, text = found["page"], found["heading"], found["text"]
	if len(text) > ANSWER_LIMIT:
		cut = text[:ANSWER_LIMIT]
		text = cut[: cut.rfind("\n\n")] if "\n\n" in cut else cut
	title = page.get("title") or "page"
	lines = [f"This is what the Frappe documentation says ({title}{f': {heading}' if heading else ''}):", "", text.strip(), "", f"Source: {page['url']}"]
	more = [{"title": p["title"], "url": p["url"]} for p in found["more"]]
	if more:
		lines += ["", "Related pages:"] + [f"- [{m['title']}]({m['url']})" for m in more]
	return {"query": topic, "page": {"title": title, "url": page["url"]}, "more": more, "answer": "\n".join(lines)}


def _web_fallback(topic: str) -> dict[str, Any]:
	"""The docs have nothing on `topic`: the web results as they are (title, link, snippet), no
	model in between. No answer, so the chat model takes over, when the search finds nothing too."""
	try:
		results = _duckduckgo(" ".join(str(topic or "").split())[:QUERY_LIMIT])
	except Exception:
		results = []
	if not results:
		return {"query": topic, "page": None, "more": [], "answer_hint": "Nothing found in the Frappe documentation or on the web."}
	lines = ["I couldn't find this in the Frappe documentation. These web results may help:", ""]
	lines += [f"- [{r['title']}]({r['url']})" + (f" — {r['snippet']}" if r.get("snippet") else "") for r in results]
	return {"query": topic, "page": None, "results": results, "source": "DuckDuckGo", "answer": "\n".join(lines)}


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


def _page_text(url: str, limit: int | None = DOCS_TEXT_LIMIT) -> str:
	"""A docs page as Markdown (docs.frappe.io serves <page>.md), without its front matter, cut to
	`limit` characters (None for all of it)."""
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
	return text if limit is None or len(text) <= limit else text[:limit].rsplit("\n", 1)[0] + "\n…"

