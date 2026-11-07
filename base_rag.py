import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import settings, client
from store import VectorStore

DISCLAIMER = ("\n\n_This is an informational tool based on the Constitution text "
              "(as amended up to the 27th Amendment, 2025). It is not legal advice._")

SYSTEM_PROMPT = """You answer questions about the Constitution of Pakistan using ONLY the numbered sources provided.
Rules:
- The sources reflect the current law, including the 27th Amendment (2025). Treat them as current.
- If the sources do not contain the answer, say so plainly. Do not use outside knowledge.
- Cite every claim like (Article 19) or (Article 175A, PDF p. 120). For the Preamble, Objectives Resolution or Schedules, cite them by name.
- If a source is marked [OMITTED], say that the article has been omitted from the Constitution.
- Amendment notes are page-level footnotes and may refer to a neighbouring article; only use them if they clearly match the article asked about.
- Be concise and plain. Quote sparingly."""

AMEND_RE = re.compile(r"amend|substitut|inserted|repeal|history|changed|omitted|added", re.I)
ART_RE = re.compile(
    r"\bArt(?:icle)?s?\.?\s+(\d+[A-Za-z]{0,3}(?:\s*(?:,|and|&|or)\s*\d+[A-Za-z]{0,3})*)\b",
    re.I,
)
NUM_RE = re.compile(r"\d+[A-Za-z]{0,3}")


# ---------------- data classes ----------------
@dataclass
class Chunk:
    id: str
    text: str
    meta: dict
    distance: float | None = None
    source: str = "semantic"          # exact | semantic | sibling


@dataclass
class Group:
    key: str                          # article_number, or section/kind for non-articles
    chunks: list[Chunk]
    via: str                          # exact | semantic
    best_distance: float | None = None

    @property
    def meta(self):
        return self.chunks[0].meta

    @property
    def is_article(self):
        return self.meta.get("kind") == "article"


@dataclass
class RAGResult:
    variant: str
    question: str
    answer: str = ""
    detected_articles: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)            # [{key, via, best_distance, chunk_ids}]
    dropped_for_budget: list = field(default_factory=list)
    cited_articles: list = field(default_factory=list)
    cited_not_retrieved: list = field(default_factory=list)  # rough flag, NOT a hallucination metric
    context: str = ""
    model: str = ""
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    error: str | None = None
    missing_articles: list = field(default_factory=list)     # detected but not in the store
    config: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)                # variant-specific trace (e.g. rewritten query)               # settings used for this run

    def display(self) -> str:
        return self.answer + DISCLAIMER

    def to_dict(self):
        return asdict(self)


# ---------------- query understanding ----------------
def detect_articles(text: str) -> list[str]:
    found = []
    for m in ART_RE.finditer(text):
        for n in NUM_RE.findall(m.group(1)):
            n = n.upper()
            if n not in found:
                found.append(n)
    return found


# ---------------- retrieval ----------------
def group_key(meta: dict) -> str:
    return meta.get("article_number") or meta.get("section") or meta.get("kind", "unknown")


def chunks_from_get(res: dict, source: str) -> list[Chunk]:
    """Flat Chroma get() result -> Chunks ordered by chunk_index."""
    chunks = [Chunk(i, d, m, None, source)
              for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
    return sorted(chunks, key=lambda c: c.meta.get("chunk_index", 0))


def exact_lookup(store, detected: list[str]) -> tuple[dict[str, Group], list[str]]:
    groups, missing = {}, []
    for num in detected:
        chunks = chunks_from_get(store.get_article(num), "exact")
        if chunks:
            groups[num] = Group(num, chunks, "exact", best_distance=-1.0)
        else:
            missing.append(num)
    return groups, missing


def semantic_search(store, query: str, n_results: int = 8,
                    skip_keys: set | None = None, expand: bool = True) -> list[Group]:
    skip_keys = skip_keys or set()
    res = store.query(query, n_results=n_results)        # nested lists -> [0]

    groups: dict[str, Group] = {}
    for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                             res["metadatas"][0], res["distances"][0]):
        key = group_key(m)
        if key in skip_keys:
            continue
        g = groups.setdefault(key, Group(key, [], "semantic", dist))
        g.chunks.append(Chunk(i, d, m, dist, "semantic"))

    for key, g in groups.items():
        if expand and g.meta.get("article_number"):      # articles: pull ALL parts
            full = chunks_from_get(store.get_article(key), "sibling")
            hit_ids = {c.id for c in g.chunks}
            for c in full:
                if c.id in hit_ids:
                    c.source = "semantic"
            if full:
                g.chunks = full
        else:                                            # annex/schedule/preamble: hits only
            g.chunks.sort(key=lambda c: c.meta.get("chunk_index", 0))

    return sorted(groups.values(), key=lambda g: g.best_distance)


def retrieve_groups(store, question: str, detected: list[str], n_results: int = 8,
                    max_groups: int = 5, query_text: str | None = None):
    """Returns (groups, missing). query_text lets variants search with a rewritten query."""
    exact, missing = exact_lookup(store, detected)
    semantic = semantic_search(store, query_text or question,
                               n_results=n_results, skip_keys=set(exact))
    return list(exact.values()) + semantic[:max_groups], missing


# ---------------- context ----------------
def label_for(g: Group) -> str:
    m = g.meta
    pages = sorted({c.meta.get("pdf_page") for c in g.chunks if c.meta.get("pdf_page") is not None})
    pg = ""
    if pages:
        pg = f"PDF p. {pages[0]}" + (f"-{pages[-1]}" if len(pages) > 1 else "")
    if g.is_article:
        title = m.get("article_title", "")
        label = f"Article {g.key}" + (f": {title}" if title else "")
        if m.get("omitted"):
            label += " [OMITTED]"
    else:
        label = "Preamble" if m.get("kind") == "preamble" else str(g.key)
    return f"{label} | {pg}".strip(" |")


def build_context_str(question: str, groups: list[Group], max_chars: int = 24000,
                      notes_fn=None, amend_re=AMEND_RE):
    want_notes = bool(amend_re.search(question))
    parts, dropped, used = [], [], 0
    for g in groups:
        body = "\n".join(c.text for c in g.chunks)
        block = f"[Source {len(parts) + 1}] {label_for(g)}\n{body}"
        if want_notes:
            if notes_fn:
                items = notes_fn(g.chunks)
                if items:
                    block += ("\nAmendment footnotes (deduplicated, oldest first; footnote numbers removed, "
                              "so a note's link to a specific clause is unknown):\n- " + "\n- ".join(items))
            else:
                notes = {c.meta["amendment_notes"] for c in g.chunks if c.meta.get("amendment_notes")}
                if notes:
                    block += "\nAmendment notes: " + " || ".join(sorted(notes))
        if used + len(block) > max_chars and parts:
            dropped.append(g.key)
            continue
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts), dropped


# ---------------- generation ----------------
def generate_answer(question: str, context: str, system_prompt: str = SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
    ]
    kwargs = dict(model=settings.deployment, messages=messages,
                  max_completion_tokens=settings.max_output_tokens)
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    resp = client.chat.completions.create(**kwargs)
    choice, u = resp.choices[0], resp.usage
    return (choice.message.content or ""), {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "finish_reason": choice.finish_reason,
    }


# ---------------- the pipeline ----------------
class BaseRAG:
    """detect -> retrieve -> build_context -> generate. Variants override single stages."""
    name = "base"
    system_prompt = SYSTEM_PROMPT            # variants can override just the prompt

    def __init__(self, store: VectorStore | None = None, n_results=8, max_groups=5,
                 max_context_chars=24000, log_path="logs/rag_runs.jsonl",log_context=False):
        self.store = store or VectorStore()
        self.n_results = n_results
        self.max_groups = max_groups
        self.max_context_chars = max_context_chars
        self.log_path = Path(log_path) if log_path else None
        self.log_context = log_context

    # ---- stages (override these) ----
    def detect(self, question):
        return detect_articles(question)

    def prepare_query(self, question):
        return question                      # hook: rewrite/translate the query for semantic search

    def retrieve(self, question, detected, query_text=None):
        if query_text is None:
            query_text = self.prepare_query(question)
        return retrieve_groups(self.store, question, detected, n_results=self.n_results,
                               max_groups=self.max_groups, query_text=query_text)

    def build_context(self, question, groups):
        return build_context_str(question, groups, max_chars=self.max_context_chars)

    def generate(self, question, context):
        return generate_answer(question, context, self.system_prompt)

    # ---- orchestration ----
    def ask(self, question: str) -> RAGResult:
        r = RAGResult(variant=self.name, question=question, model=settings.deployment)
        r.config = {"n_results": self.n_results, "max_groups": self.max_groups,
                    "max_context_chars": self.max_context_chars,
                    "temperature": settings.temperature,
                    "max_output_tokens": settings.max_output_tokens}
        t0 = time.perf_counter()
        try:
            r.detected_articles = self.detect(question)

            t = time.perf_counter()
            query_text = self.prepare_query(question)
            r.timings["prepare_query_s"] = round(time.perf_counter() - t, 3)
            r.extra["query_used"] = query_text

            t = time.perf_counter()
            groups, r.missing_articles = self.retrieve(question, r.detected_articles,
                                                       query_text=query_text)
            r.timings["retrieve_s"] = round(time.perf_counter() - t, 3)

            r.context, r.dropped_for_budget = self.build_context(question, groups)
            r.retrieved = [{"key": g.key, "via": g.via, "best_distance": g.best_distance,
                            "chunk_ids": [c.id for c in g.chunks]} for g in groups]

            if not groups:
                r.answer = "I couldn't find anything relevant in the Constitution text."
            else:
                t = time.perf_counter()
                r.answer, r.usage = self.generate(question, r.context)
                r.timings["generate_s"] = round(time.perf_counter() - t, 3)

                r.cited_articles = detect_articles(r.answer)
                have = {g.key for g in groups if g.key not in r.dropped_for_budget}
                r.cited_not_retrieved = [a for a in r.cited_articles if a not in have]
        except Exception as e:               # failures go in the log too
            r.error = f"{type(e).__name__}: {e}"
            r.answer = "Sorry, something went wrong while answering."
        r.timings["total_s"] = round(time.perf_counter() - t0, 3)
        self._log(r)
        return r

    def _log(self, r: RAGResult):
        if not self.log_path:
            return
        d = r.to_dict()
        if not self.log_context:
            d.pop("context", None)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")