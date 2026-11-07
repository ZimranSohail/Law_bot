import json
import re
from pathlib import Path
from base_rag import (BaseRAG, ART_RE, NUM_RE, AMEND_RE, detect_articles,
                      exact_lookup, semantic_search, build_context_str, generate_answer)
from config import settings, client




REWRITE_PROMPT = """Rewrite the user's question as a short search query for finding relevant articles of the Constitution of Pakistan.
- Write in formal English, using the vocabulary a constitution would use (e.g. arrest, detention, property, fundamental rights, appointment, removal).
- If the question is in another language or Roman Urdu, translate it.
- Keep any article numbers exactly as written.
- Output ONLY the search query, no explanation, no quotes."""


def strip_new_article_numbers(rewritten: str, original: str) -> str:
    """Remove article numbers the user did not write from a rewritten query."""
    allowed = set(detect_articles(original))

    def repl(m):
        nums = [n.upper() for n in NUM_RE.findall(m.group(1))]
        return m.group(0) if all(n in allowed for n in nums) else ""

    out = ART_RE.sub(repl, rewritten)
    out = re.sub(r"\b\d{1,3}[A-Za-z]{0,3}\b",
                 lambda m: m.group(0) if m.group(0).upper() in allowed else "", out)
    return re.sub(r"\s+", " ", out).strip()


class RewriteRAG(BaseRAG):
    """LLM rewrites the question into formal English for the vector search only.
    The answer step still sees the user's original question."""
    name = "rewrite"
    rewrite_prompt = REWRITE_PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def prepare_query(self, question):
        if question in self._cache:
            return self._cache[question]
        try:
            kwargs = dict(
                model=settings.deployment,
                messages=[{"role": "system", "content": self.rewrite_prompt},
                          {"role": "user", "content": question}],
                max_completion_tokens=100,
            )
            if settings.temperature is not None:
                kwargs["temperature"] = settings.temperature
            text = (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()
        except Exception as e:
            print(f"rewrite failed ({type(e).__name__}): falling back to original question")
            text = ""
        rewritten = text or question
        self._cache[question] = rewritten
        return rewritten


class RewriteStrictRAG(RewriteRAG):
    """Rewrite + strip article numbers the user didn't type + on-disk cache (reproducible evals).
    NOTE: the cache is keyed by question only. Delete the cache file if you change rewrite_prompt."""
    name = "rewrite_strict"
    cache_path = Path("eval/rewrite_cache.json")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disk = (json.loads(self.cache_path.read_text(encoding="utf-8"))
                      if self.cache_path.exists() else {})

    def prepare_query(self, question):
        if question not in self._disk:
            raw = super().prepare_query(question)
            if raw == question:                      # rewrite failed: don't cache
                return question
            self._disk[question] = raw
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._disk, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        return strip_new_article_numbers(self._disk[question], question)


def _selftest():
    assert strip_new_article_numbers("slavery article 11 Constitution", "Is slavery allowed?") == "slavery Constitution"
    assert strip_new_article_numbers("President article 41 42 43 vacancy", "President dies?") == "President vacancy"
    assert "19" in strip_new_article_numbers("Article 19 freedom", "Is Article 19 about religion?")
    print("variants selftest OK")


URDU_HINTS = {"kya","kia","kay","ke","ki","ka","hai","hain","mujhe","bina","wajah","nahi","nahin",
              "sakta","sakti","sakte","mein","aap","koi","agar","liye","batay","bataye","kaun","kon",
              "kab","kahan","kyun","kyu","hota","hoti","hote","kar","karna","kiya","giraftar",
              "mazhab","sarkari","hukumat","qanoon"}

def is_roman_urdu(text: str) -> bool:
    if re.search(r"[\u0600-\u06FF]", text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & URDU_HINTS) >= 2


class RouterRAG(RewriteStrictRAG):
    """Rewrite only Roman Urdu / Urdu-script questions; search everything else as-is."""
    name = "router"
    def prepare_query(self, question):
        return super().prepare_query(question) if is_roman_urdu(question) else question


class FusionRAG(RewriteStrictRAG):
    """Search with the original AND the rewritten query, merge by reciprocal rank fusion.
    Roman Urdu questions use the rewrite only."""
    name = "fusion"
    rrf_k = 60

    def retrieve(self, question, detected, query_text=None):
        if query_text is None:
            query_text = self.prepare_query(question)
        exact, missing = exact_lookup(self.store, detected)
        skip = set(exact)

        lists = []
        if not is_roman_urdu(question):
            lists.append(semantic_search(self.store, question, self.n_results, skip))
        if query_text != question or is_roman_urdu(question):
            lists.append(semantic_search(self.store, query_text, self.n_results, skip))

        scores, by_key = {}, {}
        for lst in lists:
            for rank, g in enumerate(lst, 1):
                scores[g.key] = scores.get(g.key, 0.0) + 1.0 / (self.rrf_k + rank)
                by_key.setdefault(g.key, g)          # keep the first list's group object
        fused = sorted(by_key.values(), key=lambda g: -scores[g.key])
        return list(exact.values()) + fused[: self.max_groups], missing


FOOTNOTE_NUM = re.compile(r"^\s*\d+\.\s*")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

def clean_notes(chunks):
    seen, items = set(), []
    for c in chunks:
        raw = c.meta.get("amendment_notes")
        if not raw:
            continue
        for part in raw.split(" | "):
            t = FOOTNOTE_NUM.sub("", part).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                items.append(t)
    def year(t):
        m = YEAR.search(t)
        return int(m.group(0)) if m else 9999
    return sorted(items, key=year)          # stable sort: ties keep page order

AMEND_RULE = """
- For questions about when an article was inserted, amended or substituted: footnotes are NOT tied to specific text. A note saying the article itself was inserted ("New Article X ins." or "Ins. ... Act") gives the original insertion, and the EARLIEST such note is the original date. Later "Subs." / "Ins." notes usually concern individual clauses, so describe them as later amendments, not as the article's insertion. If the footnotes are ambiguous, say so."""


class FusionNotesRAG(FusionRAG):
    """Fusion retrieval + cleaned, chronological footnotes + an amendment-question rule."""
    name = "fusion_notes"
    use_rule = True                          # set False to test the notes cleanup alone

    def build_context(self, question, groups):
        return build_context_str(question, groups, max_chars=self.max_context_chars,
                                 notes_fn=clean_notes)

    def generate(self, question, context):
        prompt = self.system_prompt
        if self.use_rule and AMEND_RE.search(question):
            prompt += AMEND_RULE
        return generate_answer(question, context, prompt)


AMEND_RE_WIDE = re.compile(AMEND_RE.pattern + r"|come to be|came into|became part|introduc|origin|enact|insert",
                           re.I)

def explicit_insertion_notes(key: str, chunks) -> list[str]:
    """Notes saying 'New Article <key> ins.' (tolerates '10 A'); resolves 'ibid.' by including the previous note."""
    m = re.match(r"(\d+)([A-Z]*)$", key.upper())
    if not m:
        return []
    pat = re.compile(rf"New\s+Article\s+{m.group(1)}\s*{m.group(2)}\s+ins", re.I)
    out = []
    for c in chunks:
        raw = c.meta.get("amendment_notes")
        if not raw:
            continue
        parts = raw.split(" | ")
        for i, p in enumerate(parts):
            text = FOOTNOTE_NUM.sub("", p).strip()
            if pat.search(text):
                if re.search(r"\bibid\b", text, re.I) and i > 0 and "ibid. refers to" not in text:
                    prev = FOOTNOTE_NUM.sub("", parts[i - 1]).strip()
                    if prev not in out:
                        out.append(prev)
                if text not in out:
                    out.append(text)
    return out

INSERTION_RULE = """
- If the sources include an 'Explicit insertion footnotes' section, it states how the article itself was inserted. Use it to answer questions about when or how an article was inserted or added. Other amendment notes concern later changes or individual clauses; describe them as later amendments."""


class InsertionNotesRAG(FusionRAG):
    """Fusion retrieval + widened amendment gate + deterministic 'New Article N ins.' extraction."""
    name = "fusion_insertion"

    def build_context(self, question, groups):
        ctx, dropped = build_context_str(question, groups, max_chars=self.max_context_chars,
                                         amend_re=AMEND_RE_WIDE)
        if AMEND_RE_WIDE.search(question):
            lines = []
            for g in groups:
                if g.is_article and g.key not in dropped:
                    for note in explicit_insertion_notes(g.key, g.chunks):
                        lines.append(f"[Article {g.key}] {note}")
            if lines:
                ctx += "\n\n---\n\nExplicit insertion footnotes:\n" + "\n".join(lines)
        return ctx, dropped

    def generate(self, question, context):
        prompt = self.system_prompt + (INSERTION_RULE if AMEND_RE_WIDE.search(question) else "")
        return generate_answer(question, context, prompt)



UNKNOWN_RULE = """
- If a source says 'No explicit insertion footnote found' for an article, say the provided sources do not state how or when that article was inserted. Do not attribute page-level amendment notes to it."""


class InsertionNotesV2RAG(InsertionNotesRAG):
    """Explicit insertion notes always attached for named articles; says 'unknown' when none exist."""
    name = "fusion_insertion_v2"

    def build_context(self, question, groups):
        gated = bool(AMEND_RE_WIDE.search(question))
        ctx, dropped = build_context_str(question, groups, max_chars=self.max_context_chars,
                                         amend_re=AMEND_RE_WIDE)
        lines = []
        for g in groups:
            if not g.is_article or g.key in dropped:
                continue
            named = g.via == "exact"
            if not (named or gated):
                continue
            notes = explicit_insertion_notes(g.key, g.chunks)
            for note in notes:
                lines.append(f"[Article {g.key}] {note}")
            if named and not notes and gated:
                lines.append(f"[Article {g.key}] No explicit insertion footnote found. "
                             f"The page's amendment notes may belong to neighbouring articles.")
        if lines:
            ctx += "\n\n---\n\nExplicit insertion footnotes:\n" + "\n".join(lines)
        return ctx, dropped

    def generate(self, question, context):
        prompt = self.system_prompt
        if "Explicit insertion footnotes" in context:
            prompt += INSERTION_RULE + UNKNOWN_RULE
        return generate_answer(question, context, prompt)

TONE_RULE = """
- If the question cannot be answered from the text of the Constitution at all (for example: who currently holds an office, current events, tax rates, personal legal advice), reply in at most two plain sentences: say you can only answer from the text of the Constitution of Pakistan and that it does not cover this. Do not cite articles or list sources for such a question. You may add one short sentence on what the Constitution does cover that is related.
- If the sources answer only part of the question, answer that part normally and say what is not covered."""

class DeclineToneRAG(FusionRAG):
    """Fusion retrieval + a prompt rule for short, citation-free declines."""
    name = "fusion_tone"
    system_prompt = FusionRAG.system_prompt + TONE_RULE

SCOPE_PROMPT = """Decide whether a question can be answered from the TEXT of the Constitution of Pakistan.
Answer NO only for: who currently holds an office, current events or news, current tax rates or prices, personal legal advice about someone's own case, questions about other countries, or topics clearly unrelated to the Constitution.
Answer YES for anything asking what the Constitution says, requires, allows or forbids, including rights, procedures, institutions, amendments and articles, in any language.
If the question makes a claim about the Constitution or an article, even a wrong one, answer YES. If you are unsure, answer YES.
Reply with exactly one word: YES or NO."""

DECLINE_TEXT = ("I can only answer from the text of the Constitution of Pakistan, and it does not cover this. "
                "You can ask me about a specific article, right, institution or procedure in it.")


class ScopeGateRAG(FusionRAG):
    """Fusion + a cheap scope check; out-of-scope questions get a short fixed reply, no citations."""
    name = "fusion_scope"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scope = {}

    def in_scope(self, question):
        if self.detect(question):          # names an article -> about the Constitution
            return True
        if question in self._scope:
            return self._scope[question]
        try:
            kwargs = dict(model=settings.deployment,
                          messages=[{"role": "system", "content": SCOPE_PROMPT},
                                    {"role": "user", "content": question}],
                          max_completion_tokens=10)
            if settings.temperature is not None:
                kwargs["temperature"] = settings.temperature
            text = (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip().upper()
            ok = not text.startswith("NO")
        except Exception:
            ok = True                      # fail open
        self._scope[question] = ok
        return ok

    def generate(self, question, context):
        if not self.in_scope(question):
            return DECLINE_TEXT, {"scope": "out"}
        return super().generate(question, context)

COMPLETE_RULE = """
- When an article states exceptions, provisos, time limits, conditions or separate clauses that bear on the question, include each of them briefly, even if one clause already answers the main point. Do not add anything that is not in the sources."""

class CompleteRAG(FusionRAG):
    """Fusion + a rule asking for exceptions, provisos and time limits."""
    name = "fusion_complete"
    system_prompt = FusionRAG.system_prompt + COMPLETE_RULE