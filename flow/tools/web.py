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
from frappe.utils.caching import redis_cache

from flow.lib.tool import LayaRoute, tool

SEARCH_URL = "https://html.duckduckgo.com/html/"
DOCS_HOST = "docs.frappe.io"
DOCS_SEARCH_URL = f"https://{DOCS_HOST}/api/method/wiki.frappe_wiki.doctype.wiki_document.search.search"
DOCS_MORE_LIMIT = 4
DOCS_OPTIONS = 6  # sections a model may choose between
NAMED_OPTIONS = 2  # of them, from the page named like a DocType of the question
OPTION_PREVIEW = 110  # characters of a section shown beside its heading when choosing
DOCS_CANDIDATES = 6  # pages docs_howto reads to find the section that fits best
MAX_OPENING_WORDS = 3  # words of a question that may be left off its search ("how do I")
PAGES_PER_SEARCH = 2
MIN_DOCTYPE_WORD = 4
DOCTYPE_QUERIES = 2
MIN_VARIANT_WORDS = 2
# The model reads the page in a small context window; the rest is a link away.
DOCS_TEXT_LIMIT = 3500
RESULTS_LIMIT = 5
TIMEOUT_SECONDS = 10
QUERY_LIMIT = 200
# Web results are kept to these sources (host suffix, path prefix); a site can set its own list in
# `flow_web_search_domains` ("host" or "host/path" strings). Anything else is not about Frappe.
DEFAULT_WEB_SOURCES = ("frappe.io", "frappe.school", "frappecloud.com", "erpnext.com", "github.com/frappe")
RAW_RESULTS_LIMIT = 20  # read from the search, before the sources filter
# Said by the scope check when a message is not about this site or ERPNext / Frappe at all.
OUT_OF_SCOPE = (
	"I can only help with questions about this site and ERPNext / Frappe. "
	"Ask me about your records, roles, or how to do something in ERPNext."
)
# Said when the docs and the Frappe / ERPNext sources have nothing on a question: it may well be an
# ERP question, so it is not called off-topic, and the chat model must not improvise an answer.
NOT_IN_DOCS = (
	"I couldn't find this in the Frappe documentation. Try asking in different words, "
	"or ask about your records, roles or settings."
)


@tool
def web_search(
	query: Annotated[str, 'Search words, e.g. "ERPNext give a role permission on a doctype". No record IDs or personal data.'],
) -> dict[str, Any]:
	"""Search Frappe / ERPNext sources on the web (docs, forum, GitHub, Frappe School), only when no
	site tool or docs_howto can answer. It does not search the rest of the web: anything unrelated
	to Frappe / ERPNext returns nothing. Answer from the snippets and name the source links."""
	if _turned_off():
		raise ValueError("Web search is turned off on this site.")
	query = " ".join(str(query or "").split())[:QUERY_LIMIT]
	if not query:
		raise ValueError("Give the search words.")
	results = _scoped_search(query)
	if not results:
		return {
			"query": query,
			"results": [],
			"answer_hint": f"Nothing found in Frappe / ERPNext sources. Reply with exactly this and nothing else: {NOT_IN_DOCS}",
		}
	return {"query": query, "results": results, "source": "DuckDuckGo"}


def _web_sources() -> tuple[tuple[str, str], ...]:
	"""(host suffix, path prefix) of every source web results may come from."""
	configured = frappe.conf.get("flow_web_search_domains")
	entries = configured if isinstance(configured, (list, tuple)) and configured else DEFAULT_WEB_SOURCES
	out = []
	for entry in entries:
		host, _, path = str(entry).strip().lower().partition("/")
		if host:
			out.append((host, f"/{path}" if path else ""))
	return tuple(out)


def _is_web_source(url: str) -> bool:
	parsed = urlparse(url)
	host = (parsed.hostname or "").lower()
	return any(
		(host == source or host.endswith(f".{source}")) and parsed.path.lower().startswith(prefix)
		for source, prefix in _web_sources()
	)


def _scoped_search(query: str) -> list[dict[str, str]]:
	"""Web results for `query` from the Frappe / ERPNext sources only, so a question that has
	nothing to do with them (the weather, the news) finds nothing instead of an answer."""
	sites = " OR ".join(f"site:{host}{path}" for host, path in _web_sources())
	found = _duckduckgo(f"{query} ({sites})", limit=RAW_RESULTS_LIMIT)
	return [r for r in found if _is_web_source(r["url"]) and _mentions_query(r, query)][:RESULTS_LIMIT]


def _mentions_query(result: dict[str, str], query: str) -> bool:
	"""Whether a result's title, snippet or link holds at least half of the query's longer words
	(5+ letters, matched on their first five). A search limited to some sites still returns pages
	of those sites for any words, so this is what keeps "the weather today" from finding ERPNext."""
	words = {w[:5] for w in re.findall(r"[a-z]+", query.lower()) if len(w) >= 5}
	if not words:
		return True
	text = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}".lower()
	return sum(w in text for w in words) * 2 >= len(words)


def _turned_off() -> bool:
	return str(frappe.conf.get("flow_web_search_enabled")).strip().lower() in ("0", "false", "no")


def _duckduckgo(query: str, limit: int = RESULTS_LIMIT) -> list[dict[str, str]]:
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
		if len(results) >= limit:
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
# A section answers a question only when it holds at least this share of the question's own words of
# 4+ letters, counted one each: a rare word alone carries nearly all the score, so "explain recursion
# in simple words" would otherwise be answered by any page that merely mentions "recursion".
# (Short words such as "how", "add" and "new" are left out: pages often lack them.)
MIN_WORD_SHARE = 0.5
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


def rank_sections(pages: list[tuple[Any, str]], topic: str) -> list[tuple[float, Any, str | None, str]]:
	"""(score, page, heading, body) of every part, across `pages` of (page, text), that fits `topic`,
	best first. A topic word counts for more the fewer parts mention it ("how", "the" and "to" are
	in nearly all, so count for nothing), and three times as much in a heading as in the body. Ties
	keep search order. Parts holding too few of the topic's words are left out (MIN_WORD_SHARE)."""
	words = {w[:5] for w in re.findall(r"[a-z]+", topic.lower())}
	parts = [(page, heading, body) for page, text in pages for heading, body in sections(text)]
	if not parts:
		return []
	haystacks = [f"{heading or ''}\n{body}".lower() for _page, heading, body in parts]
	weight = {w: math.log((len(parts) + 1) / (sum(w in h for h in haystacks) + 1)) for w in words}
	long_words = {w for w in words if len(w) >= 4} or words
	ranked = []
	for (page, heading, body), haystack in zip(parts, haystacks):
		score = sum(
			weight[w] * (3 if heading and w in heading.lower() else 1) for w in words if w in haystack
		)
		shared = sum(w in haystack for w in long_words) / len(long_words)
		if score > 0 and shared >= MIN_WORD_SHARE:
			ranked.append((score, page, heading, body))
	return sorted(ranked, key=lambda r: -r[0])


def pick_section(pages: list[tuple[Any, str]], topic: str) -> tuple[Any, str | None, str, float]:
	"""(page, heading, body, score) of the part of `pages` that fits `topic` best; when nothing
	matches, the first page's opening with score 0."""
	ranked = rank_sections(pages, topic)
	if not ranked:
		return pages[0][0], None, "", 0.0
	score, page, heading, body = ranked[0]
	return page, heading, body, score


def _candidate_pages(topic: str) -> list[dict[str, str]]:
	"""The pages to read for `topic`. docs.frappe.io's search wants every word on a page, so a
	whole question often finds nothing: "how do I import customers" fails on "how do I" and a
	record type named in it pulls in pages about that record, not the task. So the topic is
	searched as asked, with the opening words of the question left off ("how do I"), and without
	the words that name a DocType on this site; the first page or two of each are read."""
	from concurrent.futures import ThreadPoolExecutor

	words = topic.split()
	named_words = _doctype_words(words)  # one lookup, not one per search
	queries = []
	for skipped in range(MAX_OPENING_WORDS + 1):
		rest = words[skipped:]
		if len(rest) < MIN_VARIANT_WORDS:
			break
		queries.append(" ".join(rest))
		tails = [w for w in rest if w not in named_words]
		if MIN_VARIANT_WORDS <= len(tails) < len(rest):
			queries.append(" ".join(tails))
	doctypes = _doctype_queries(words, named_words)
	queries = list(dict.fromkeys(queries + doctypes)) or [topic]

	def fetch(query: str) -> list[dict[str, Any]]:
		try:
			return _fetch_docs(query)
		except ValueError:
			return []

	with ThreadPoolExecutor(max_workers=len(queries)) as pool:
		fetched = list(pool.map(fetch, queries))
	installed, preferred = _installed_app_words(), _preferred_apps(topic)
	ranked = [_rank_pages(hits, installed, preferred) for hits in fetched]
	found: dict[str, dict[str, str]] = {}
	for rank in range(PAGES_PER_SEARCH):
		for pages in ranked:
			for page in pages[rank : rank + 1]:
				found.setdefault(page["url"], page)
	# A page titled like a DocType of the site that a topic word is part of is that feature's own page.
	named = {name.lower() for name in doctypes}
	pages = [{**p, "named": p["title"].lower() in named} for p in found.values()]
	return sorted(pages, key=lambda p: not p["named"])[:DOCS_CANDIDATES]


def _doctype_queries(words: list[str], named_words: set[str]) -> list[str]:
	"""Names of this site's DocTypes that contain a word of the topic ("import" -> "Data Import"):
	the docs page of a feature is titled like its DocType, and the page does not repeat the
	words around the DocType in a question. The shortest names are the most general."""
	names: list[str] = []
	for word in dict.fromkeys(w for w in words if len(w) >= MIN_DOCTYPE_WORD and w not in named_words):
		found = frappe.get_all("DocType", filters={"name": ["like", f"%{word}%"], "istable": 0}, pluck="name")
		names += sorted(found, key=len)[:DOCTYPE_QUERIES]
	return list(dict.fromkeys(names))


def _doctype_words(words: list[str]) -> set[str]:
	"""The words ("customers", "invoice") that are the name of a DocType on this site."""
	candidates = {w: {w.title(), (w[:-1] if len(w) > 3 and w.endswith("s") else w).title()} for w in set(words)}
	known = set(frappe.get_all("DocType", filters={"name": ["in", sorted({n for names in candidates.values() for n in names})]}, pluck="name"))
	return {w for w, names in candidates.items() if names & known}


def _answering_section(topic: str) -> dict[str, Any] | None:
	"""The best (page, heading, text) for `topic` among the top few pages the docs search returns:
	the search ranks whole pages by loose matching, so the page whose own section fits the words
	best is picked instead of simply the first. None when the search finds nothing, or when no
	section mentions any of the words (the docs don't cover it)."""
	from concurrent.futures import ThreadPoolExecutor

	found = _candidate_pages(" ".join(str(topic or "").split())[:QUERY_LIMIT])
	if not found:
		return None
	with ThreadPoolExecutor(max_workers=len(found)) as pool:
		texts = list(pool.map(lambda p: _page_text(p["url"], None) or p["snippet"], found))
	pages = list(zip(found, texts))
	ranked = rank_sections(pages, topic)
	if not ranked:
		return None
	# A feature's own page ("Data Import") comes first, but pages that merely mention the words stay
	# among the options: the question may be about something that page only touches on
	# ("a discount on a sales order" is on its own page, not the Sales Order one).
	named = [r for r in rank_sections([(p, t) for p, t in pages if p.get("named")], topic)][:NAMED_OPTIONS]
	seen, chosen = set(), []
	for row in named + ranked:  # the same section can rank in both lists
		key = (row[1]["url"], row[2])
		if key not in seen:
			seen.add(key)
			chosen.append(row)
	chosen = chosen[:DOCS_OPTIONS]
	return {"sections": [{"page": page, "heading": heading, "text": text} for _score, page, heading, text in chosen]}


def _format_section(section: dict[str, Any]) -> str:
	"""A docs section as the answer shown to the user: its text (cut at a paragraph) and its link."""
	text = section["text"]
	if len(text) > ANSWER_LIMIT:
		cut = text[:ANSWER_LIMIT]
		text = cut[: cut.rfind("\n\n")] if "\n\n" in cut else cut
	return "\n".join([text.strip(), "", f"Source: {section['page']['url']}"])


@tool(final_answer=True, laya=LayaRoute("HOWTO", lambda text, signals: topic_arguments(text)))
def docs_howto(
	topic: Annotated[str, 'Search words for the docs, e.g. "convert lead to deal".'],
) -> dict[str, Any]:
	"""Answer a how-to question from the Frappe / ERPNext docs (docs.frappe.io): the part of the
	best page that matches, and its link. When the docs have nothing on it, the web search results
	instead. Use frappe_docs instead when you want to read the page yourself."""
	if _turned_off():
		raise ValueError("Looking things up online is turned off on this site.")
	if not any(c.isalpha() for c in str(topic or "")):
		raise ValueError("Give the search words.")
	found = _answering_section(topic)
	if not found:
		return _web_fallback(topic)
	best = found["sections"][0]
	# The sections that fit the words best, even a single one: a model that reads the question
	# picks among them, or says none answers it (Agent._pick_option); without one the best match
	# is the answer.
	return {
		"query": topic,
		"page": {"title": best["page"].get("title") or "page", "url": best["page"]["url"]},
		"answer": _format_section(best),
		"options": [{"label": _option_label(s), "answer": _format_section(s)} for s in found["sections"]],
	}


def _option_label(section: dict[str, Any]) -> str:
	"""One line naming a section for the model: its page, heading and opening words."""
	opening = " ".join(section["text"].split())[:OPTION_PREVIEW]
	heading = f" > {section['heading'].strip('*# ')}" if section["heading"] else ""
	return f"{section['page'].get('title') or 'page'}{heading}: {opening}"


def _web_fallback(topic: str) -> dict[str, Any]:
	"""The docs have nothing on `topic`: the Frappe / ERPNext web results as they are (title, link,
	snippet), no model in between. When the search works and finds nothing, the answer says the docs
	have nothing on it; when the search itself fails there is no answer,
	so the chat model takes over."""
	try:
		results = _scoped_search(" ".join(str(topic or "").split())[:QUERY_LIMIT])
	except Exception:
		return {"query": topic, "page": None, "answer_hint": "The Frappe documentation and web search can't be reached right now."}
	if not results:
		return {"query": topic, "page": None, "answer": NOT_IN_DOCS}
	lines = ["I couldn't find this in the Frappe documentation. These web results may help:", ""]
	lines += [f"- [{r['title']}]({r['url']})" + (f" — {r['snippet']}" if r.get("snippet") else "") for r in results]
	return {"query": topic, "page": None, "results": results, "source": "DuckDuckGo", "answer": "\n".join(lines)}


def _docs_search(query: str) -> list[dict[str, str]]:
	"""Pages docs.frappe.io's own search finds (it is a Frappe Wiki site), best first: pages of
	the apps this site runs before others (the docs also cover Frappe Cloud, Books…), and current
	pages, with old versioned copies (…/v12/…) left out whenever a current page exists."""
	return _rank_pages(_fetch_docs(query), _installed_app_words(), _preferred_apps(query))


def _fetch_docs(query: str) -> list[dict[str, Any]]:
	"""The raw hits of one docs search. Needs no Frappe context, so searches can run in threads."""
	import requests

	try:
		response = requests.get(
			DOCS_SEARCH_URL,
			params={"query": query},
			headers={"User-Agent": "Mozilla/5.0 (compatible; FlowAssistant/1.0)"},
			timeout=TIMEOUT_SECONDS,
		)
		response.raise_for_status()
		return (response.json().get("message") or {}).get("results") or []
	except (requests.RequestException, ValueError) as e:
		raise ValueError(f"The Frappe documentation can't be reached right now ({type(e).__name__}).") from e


def _rank_pages(found: list[dict[str, Any]], installed: set[str], preferred: list[set[str]]) -> list[dict[str, str]]:
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
				"rank": (parts[0] not in installed, _app_order(parts[0], preferred), any(re.fullmatch(r"v\d+", p) for p in parts), i),
			}
		)
	pages.sort(key=lambda p: p.pop("rank"))
	return _current_only(_installed_only(pages, installed))


PREFERRED_DOCTYPES = 20


def _preferred_apps(query: str) -> list[set[str]]:
	"""The installed apps this site actually uses for `query`, most used first, as sets of the words
	docs.frappe.io uses for them. Found from the site's own DocTypes whose names contain a query word
	("lead" -> "Lead" of one app, "CRM Lead" of another), ranked by how many records they hold, so a
	topic that two apps cover is answered from the docs of the app whose data is here."""
	words = {w for w in re.findall(r"[a-z]+", str(query).lower()) if len(w) >= 4}
	if not words:
		return []
	try:
		doctypes = []
		for word in sorted(words):
			doctypes += frappe.get_all(
				"DocType",
				filters={"name": ["like", f"%{word}%"], "istable": 0, "issingle": 0},
				fields=["name", "module"],
				limit=PREFERRED_DOCTYPES,
			)
		app_of = dict(frappe.get_all("Module Def", fields=["name", "app_name"], as_list=True))
		used: dict[str, int] = {}
		# One count per DocType, at most PREFERRED_DOCTYPES in all, and cached: a docs lookup is a request.
		for doctype in list({d.name: d for d in doctypes}.values())[:PREFERRED_DOCTYPES]:
			app = app_of.get(doctype.module)
			if app:
				used[app] = used.get(app, 0) + _record_count(doctype.name)
	except Exception:
		# Only the ranking is lost (search order decides); say so rather than fail silently.
		frappe.logger("flow", allow_site=True).warning("Could not rank docs by the apps the site uses", exc_info=True)
		return []
	return [_app_words(app) for app, count in sorted(used.items(), key=lambda kv: -kv[1]) if count]


@redis_cache(ttl=3600)
def _record_count(doctype: str) -> int:
	"""Records of a DocType, for ordering apps only (never shown). Cached for an hour: it only has to
	tell an app the site uses from one it merely has installed."""
	return frappe.db.count(doctype)


def _app_words(app: str) -> set[str]:
	title = (frappe.get_hooks("app_title", app_name=app) or [app])[0]
	return {app.lower(), *(w.lower() for w in title.split())}


def _app_order(section: str, preferred: list[set[str]]) -> int:
	"""Position of the docs section's app among the apps the site uses for the topic."""
	return next((i for i, words in enumerate(preferred) if section in words), len(preferred))


def _installed_only(pages: list[dict[str, str]], installed: set[str]) -> list[dict[str, str]]:
	"""Only pages of products installed on this site: the docs also cover Frappe Books, Frappe Cloud
	and others, whose pages answer a different product's question ("double entry" -> Books journal
	entries). Without any page of an installed product there is no docs answer."""
	return [p for p in pages if urlparse(p["url"]).path.strip("/").split("/")[0].lower() in installed]


def _is_versioned(url: str) -> bool:
	return any(re.fullmatch(r"v\d+", part) for part in url.lower().split("/"))


def _current_only(pages: list[dict[str, str]]) -> list[dict[str, str]]:
	"""Drop the old versioned copies (…/v12/…) when current pages exist: their steps can be out of
	date and they only add noise beside the current page. Kept when they are all there is."""
	current = [p for p in pages if not _is_versioned(p["url"])]
	return current or pages


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

