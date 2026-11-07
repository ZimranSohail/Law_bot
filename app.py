import html
import os
import re
import threading
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Constitution of Pakistan · Law_bot", page_icon="⚖️",
                   layout="centered", initial_sidebar_state="collapsed")

MAX_QUESTIONS_PER_SESSION = 15
DAILY_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "150"))
REPO_URL = "https://github.com/ZimranSohail/Law_bot"
MAX_SRC_CHARS = 3000

EXAMPLE_GROUPS = {
    "Fundamental rights": [
        "Can the police arrest me without telling me why?",
        "Is slavery or forced labour allowed?",
    ],
    "Courts and offices": [
        "Who appoints the Chief Justice?",
        "What is Article 184?",
    ],
    "Amendments": [
        "When was Article 25A inserted?",
    ],
    "In Roman Urdu": [
        "Kya mujhe bina wajah batay giraftar kiya ja sakta hai?",
    ],
}


# ------------------------------------------------------------------ limits
@st.cache_resource
def usage_counter():
    return {"day": time.strftime("%Y-%m-%d"), "n": 0, "lock": threading.Lock()}


def global_allow():
    c = usage_counter()
    with c["lock"]:
        today = time.strftime("%Y-%m-%d")
        if c["day"] != today:
            c["day"], c["n"] = today, 0
        if c["n"] >= DAILY_LIMIT:
            return False
        c["n"] += 1
        return True


@st.cache_resource(show_spinner="Loading the Constitution index...")
def load_rag():
    if not Path("chroma_db").exists():            # fresh host without a committed store: build it
        from ingest import ConstitutionIngestor, PDFExtractor, TextCleaner, ArticleChunker
        from store import VectorStore
        ConstitutionIngestor(PDFExtractor(), TextCleaner(), ArticleChunker(), VectorStore()).run(
            "constitution_of_Pakistan.pdf", reset=True)
    from variants import InsertionNotesV2RAG

    class AppRAG(InsertionNotesV2RAG):
        cache_path = Path("logs/app_rewrite_cache.json")   # keep visitors' questions out of eval/

    return AppRAG()


# ------------------------------------------------------------------ design
def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
.lb-hero{text-align:center;padding:.4rem 0 0}
.lb-mark svg{width:58px;height:58px;color:%ACCENT%}
.lb-urdu{font-family:'Noto Nastaliq Urdu','Noto Naskh Arabic','Segoe UI',serif;font-size:1.7rem;line-height:2.3;color:%ACCENT%;direction:rtl;margin:0}
.lb-title{font-size:2.1rem;font-weight:700;letter-spacing:.01em;line-height:1.15;margin:0}
.lb-sub{opacity:.75;margin:.35rem 0 .7rem}
.lb-pill{display:inline-block;border:1px solid color-mix(in srgb,currentColor 28%,transparent);border-radius:999px;padding:.12rem .8rem;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;opacity:.85}
.lb-orn{display:flex;align-items:center;gap:.8rem;width:72%;margin:1.1rem auto .4rem;color:%BRASS%}
.lb-orn::before,.lb-orn::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,transparent,%BRASS%)}
.lb-orn::after{transform:scaleX(-1)}
.lb-orn span{font-size:.65rem}
.lb-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:%ACCENT%;margin:1rem 0 .3rem}
.lb-meta{font-size:.78rem;opacity:.7;margin-top:.9rem}
.lb-src{white-space:pre-wrap;border-left:3px solid %ACCENT%;border-radius:0 6px 6px 0;padding:.5rem .9rem;margin:.25rem 0 .5rem;background:color-mix(in srgb,currentColor 5%,transparent);font-size:.9rem;line-height:1.6}
.lb-foot{text-align:center;font-size:.78rem;opacity:.65;margin:2rem 0 .5rem}
</style>"""

MARK_SVG = ('<svg viewBox="0 0 64 64" aria-hidden="true"><defs><mask id="lbm">'
            '<rect width="64" height="64" fill="#fff"/><circle cx="38" cy="30" r="17" fill="#000"/>'
            '</mask></defs><circle cx="30" cy="32" r="22" fill="currentColor" mask="url(#lbm)"/>'
            '<polygon fill="currentColor" points="46,21 47.65,25.73 52.66,25.84 48.66,28.87 '
            '50.11,33.66 46,30.8 41.89,33.66 43.34,28.87 39.34,25.84 44.35,25.73"/></svg>')

HERO = ('<div class="lb-hero"><div class="lb-mark">%SVG%</div>'
        '<p class="lb-urdu">آئینِ پاکستان</p>'
        '<p class="lb-sub">You have rights. Ask where they are written.</p>'
        '<div class="lb-title">Constitution of Pakistan</div>'
        '<p class="lb-sub">Ask a question. Get an answer cited from the text.</p>'
        '<span class="lb-pill">Unofficial study tool · not legal advice</span>'
        '<div class="lb-orn"><span>◆</span></div></div>')

FOOT = ('<div class="lb-foot">Unofficial study tool · not legal advice · '
        'text of the Constitution as amended up to the 27th Amendment (2025)</div>')


def inject_design():
    dark = is_dark()
    css = (CSS.replace("%ACCENT%", "#7BD3A0" if dark else "#01411C")
              .replace("%BRASS%", "#C9A24B" if dark else "#A8801F"))
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(HERO.replace("%SVG%", MARK_SVG), unsafe_allow_html=True)


# ------------------------------------------------------------------ answer rendering
def split_sources(context):
    """Turn the context string back into [{label, meta, text}] for display."""
    out = []
    for block in (context or "").split("\n\n---\n\n"):
        if block.startswith("[Source"):
            head, _, body = block.partition("\n")
            label = head.split("]", 1)[-1].strip()
            meta, _, text = body.partition("\n\n")
            out.append({"label": label, "meta": meta.strip(), "text": (text or meta).strip()})
        elif block.startswith("Explicit insertion footnotes"):
            out.append({"label": "Insertion footnotes", "meta": "",
                        "text": block.split(":", 1)[-1].strip()})
    return out


def article_of(label):
    m = re.match(r"Article\s+(\w+)", label)
    return m.group(1).upper() if m else None


def chip_groups(r):
    cited_nums = set(r.cited_articles)
    cited, also = [f"Article {a}" for a in r.cited_articles], []
    for g in r.retrieved:
        k = g["key"]
        if k[:1].isdigit():
            name, is_cited = f"Article {k}", k in cited_nums
        else:
            name, is_cited = k, k in r.answer          # annex / preamble / schedule cited by name
        if is_cited and name not in cited:
            cited.append(name)
        elif not is_cited and name not in also:
            also.append(name)
    return cited, also[:6]


def render(r):
    st.markdown(r.answer)
    cited, also = chip_groups(r)
    if cited:
        st.markdown("**Cited**  " + "  ".join(f":green-badge[{c}]" for c in cited))
    if also:
        st.markdown("Also consulted  " + "  ".join(f":gray-badge[{c}]" for c in also))

    sources = split_sources(r.context)
    cited_nums = set(r.cited_articles)
    sources.sort(key=lambda s: 0 if article_of(s["label"]) in cited_nums else 1)
    title = f"Read the text behind this answer ({len(sources)})" if sources else "How this answer was found"
    with st.expander(title):
        for s in sources:
            text = s["text"]
            if len(text) > MAX_SRC_CHARS:
                text = text[:MAX_SRC_CHARS] + "\n[...shortened here; the full text is in the PDF]"
            st.markdown(f"**{s['label']}**")
            if s["meta"]:
                st.markdown(f'<div class="lb-meta">{html.escape(s["meta"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="lb-src">{html.escape(text)}</div>', unsafe_allow_html=True)
        st.caption(f"Detected in your question: {', '.join(r.detected_articles) or 'no article number'} · "
                   f"{r.timings.get('total_s', 0):.1f}s · {r.usage.get('total_tokens', 0)} tokens · "
                   f"model {r.model}")
        if r.error:
            st.error(r.error)
    st.caption("Informational only, not legal advice.")


# ------------------------------------------------------------------ page
inject_design()

if "history" not in st.session_state:
    st.session_state.history, st.session_state.count = [], 0

with st.sidebar:
    st.markdown("### About")
    st.write("Answers come only from the text of the Constitution of Pakistan, as amended up to the "
             "27th Amendment (2025). Each answer cites the articles it used, and you can read the "
             "source text under it.")
    st.markdown("### Good to know")
    st.write("Each question stands alone, with no memory of earlier ones. Answers can be incomplete or "
             "wrong, so check the official text. You can ask in English or Roman Urdu.")
    st.caption(f"Questions this session: {st.session_state.count}/{MAX_QUESTIONS_PER_SESSION}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("[Source code on GitHub](https://github.com/ZimranSohail/Law_bot)")

landing = st.empty()
if not st.session_state.history:
    with landing.container():
        st.markdown('<div class="lb-eyebrow" style="text-align:center">Where would you like to begin?</div>',
                    unsafe_allow_html=True)
        for group, qs in EXAMPLE_GROUPS.items():
            st.markdown(f'<div class="lb-eyebrow">{html.escape(group)}</div>', unsafe_allow_html=True)
            for i, q in enumerate(qs):
                if st.button(q, key=f"ex_{group}_{i}", use_container_width=True):
                    st.session_state.pending = q

for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        render(r)

question = st.chat_input("Ask about the Constitution of Pakistan") or st.session_state.pop("pending", None)
if question:
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Session limit reached. Reload the page to continue.")
    elif not global_allow():
        st.warning("The demo has reached its daily limit. Please try again tomorrow.")
    else:
        landing.empty()
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the Constitution..."):
                r = load_rag().ask(question[:500])
            render(r)
        st.session_state.history.append((question, r))
        st.session_state.count += 1

st.markdown(FOOT, unsafe_allow_html=True)
