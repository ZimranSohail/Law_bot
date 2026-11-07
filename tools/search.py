"""Search layer for the Constitution of Pakistan RAG bot."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INC = ["documents", "metadatas"]
NUM = r"\d{1,3}[A-Za-z]{0,2}"
ART_RE = re.compile(
    rf"\b(?:articles?|arts?\.?)\s+({NUM}(?:\s*(?:,|and|&|to|through|-|–)\s*{NUM})*)", re.I)
MAX_RANGE = 20  # ignore silly ranges like "articles 1 to 280"
AMEND_RE = re.compile(r"\b(amend\w*|inserted|substituted|omitted|repeal\w*|added|"
                      r"when was|which act|history)\b", re.I)
PURE_LOOKUP = re.compile(
    rf"^\s*(?:what\s+(?:does|do|is|are)|explain|show(?:\s+me)?|read|tell me about|give me|text of|compare)?"
    rf"\s*(?:the\s+)?(?:articles?|arts?\.?)\s+{NUM}(?:\s*(?:,|and|&|to|through|-|–)\s*{NUM})*"
    rf"(?:\s+(?:say|says|mean|provide|provides))?\s*[?.]?\s*$", re.I)

def is_pure_lookup(q: str) -> bool:
    nums, ranges = parse_articles(q)
    return bool(nums or ranges) and bool(PURE_LOOKUP.match(q))

def parse_articles(q: str):
    """-> (list of article_number strings, list of (lo, hi) int ranges)"""
    nums, ranges = [], []
    for m in ART_RE.finditer(q):
        span = m.group(1)
        toks = [t.upper() for t in re.findall(NUM, span)]
        is_range = re.search(r"\bto\b|\bthrough\b|[-–]", span, re.I) and len(toks) == 2
        if is_range:
            lo, hi = (int(re.match(r"\d+", t).group()) for t in toks)
            if 0 <= hi - lo <= MAX_RANGE:
                ranges.append((lo, hi))
                continue
        nums += [t for t in toks if t not in nums]
    return nums, ranges


def wants_amendments(q: str) -> bool:
    return bool(AMEND_RE.search(q))


@dataclass
class Hit:
    key: str                 # "19", "175A", "Preamble", "First Schedule", ...
    kind: str
    title: str
    text: str                # all parts, in order
    printed_pages: list
    pdf_pages: list
    amendment_notes: str     # PAGE-level: may describe a neighbouring article
    omitted: bool
    distance: float | None
    source: str              # "exact" | "semantic"


@dataclass
class SearchResponse:
    hits: list[Hit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # requested articles not in the collection


class ConstitutionSearcher:
    def __init__(self, store, use_prefix: bool = True):
        self.store = store
        self.col = store.collection
        self.use_prefix = use_prefix

    @staticmethod
    def _key(meta):
        return meta.get("article_number") or meta.get("section") or meta.get("kind")

    @staticmethod
    def _build_hit(key, got, distance, source) -> Hit:
        rows = sorted(zip(got["documents"], got["metadatas"]),
                      key=lambda r: r[1].get("chunk_index", 0))
        metas = [m for _, m in rows]
        notes = []
        for m in metas:
            n = m.get("amendment_notes")
            if n and n not in notes:
                notes.append(n)
        first = metas[0]
        return Hit(
            key=key, kind=first.get("kind"),
            title=first.get("article_title") or first.get("section") or first.get("kind"),
            text="\n\n".join(d for d, _ in rows),
            printed_pages=sorted({m["printed_page"] for m in metas if m.get("printed_page")}),
            pdf_pages=sorted({m["pdf_page"] for m in metas if "pdf_page" in m}),
            amendment_notes=" || ".join(notes),
            omitted=any(m.get("omitted") for m in metas),
            distance=distance, source=source)

    def _fetch_group(self, meta):
        where = ({"article_number": meta["article_number"]} if "article_number" in meta
                 else {"section": meta["section"]})
        return self.col.get(where=where, include=INC)

    def exact_hits(self, nums, ranges):
        hits, missing = [], []
        for n in nums:
            got = self.col.get(where={"article_number": n}, include=INC)
            if got["ids"]:
                hits.append(self._build_hit(n, got, None, "exact"))
            else:
                missing.append(n)
        for lo, hi in ranges:
            got = self.col.get(where={"$and": [{"article_num": {"$gte": lo}},
                                               {"article_num": {"$lte": hi}}]}, include=INC)
            buckets = {}
            for d, m in zip(got["documents"], got["metadatas"]):
                b = buckets.setdefault(m["article_number"], {"documents": [], "metadatas": []})
                b["documents"].append(d)
                b["metadatas"].append(m)
            for k in sorted(buckets, key=lambda k: (buckets[k]["metadatas"][0]["article_num"], k)):
                hits.append(self._build_hit(k, buckets[k], None, "exact"))
        return hits, missing

    def rank_keys(self, q, n_fetch=30, where=None):
        """Distinct articles/sections ranked by best chunk distance (cheap; used by evals)."""
        text = (QUERY_PREFIX + q) if self.use_prefix else q
        res = self.store.query(text, n_results=n_fetch, where=where)
        best = {}
        for meta, d in zip(res["metadatas"][0], res["distances"][0]):
            best.setdefault(self._key(meta), (d, meta))  # results arrive sorted, first = best
        return [(k, d, m) for k, (d, m) in best.items()]

    def search(self, q, n_fetch=40, n_articles=8, max_stubs=2) -> SearchResponse:
        nums, ranges = parse_articles(q)
        hits, missing = self.exact_hits(nums, ranges)
        if is_pure_lookup(q) and (hits or missing):
            return SearchResponse(hits, missing)
        seen = {h.key for h in hits}
        real, stubs = [], []
        for k, d, meta in self.rank_keys(q, n_fetch):
            if k in seen:
                continue
            is_stub = bool(meta.get("omitted"))
            bucket, limit = (stubs, max_stubs) if is_stub else (real, n_articles)
            if len(bucket) >= limit:
                if len(real) >= n_articles and len(stubs) >= max_stubs:
                    break
                continue
            bucket.append(self._build_hit(k, self._fetch_group(meta), d, "semantic"))
            seen.add(k)
        return SearchResponse(hits + real + stubs, missing)


def format_context(resp: SearchResponse, include_notes: bool = False) -> str:
    """Plain-text context block for the LLM prompt."""
    blocks = []
    for h in resp.hits:
        label = f"Article {h.key}" if h.kind == "article" else h.key
        pages = ", ".join(h.printed_pages) or "n/a"
        flag = "  [OMITTED FROM THE CURRENT CONSTITUTION]" if h.omitted else ""
        head = f"=== {label} | {h.title} | printed page(s): {pages} | match: {h.source}{flag} ==="
        body = h.text
        if include_notes and h.amendment_notes:
            body += ("\n\n[Footnotes from the same PDF page(s). They may describe a neighbouring "
                     "article, so only attribute a note to this article if its wording clearly "
                     f"refers to it:]\n{h.amendment_notes}")
        blocks.append(f"{head}\n{body}")
    if resp.missing:
        blocks.append("=== NOT FOUND === The user asked about: " + ", ".join(
            f"Article {m}" for m in resp.missing) + ". No such article exists in this text.")
    return "\n\n".join(blocks)