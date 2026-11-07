from pathlib import Path
import os, time, threading
import streamlit as st

st.set_page_config(page_title="Law_bot: Constitution of Pakistan", page_icon="⚖️")

MAX_QUESTIONS_PER_SESSION = 15
EXAMPLES = [
    "Can the police arrest me without telling me why?",
    "When was Article 25A inserted?",
    "What is Article 184?",
    "Kya mujhe bina wajah batay giraftar kiya ja sakta hai?",
    "Who appoints the Chief Justice?",
]
DAILY_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "150"))


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
    if not Path("chroma_db").exists():            # first start on a fresh host: build the index
        from ingest import ConstitutionIngestor, PDFExtractor, TextCleaner, ArticleChunker
        from store import VectorStore
        ConstitutionIngestor(PDFExtractor(), TextCleaner(), ArticleChunker(), VectorStore()).run(
            "constitution_of_Pakistan.pdf", reset=True)
    from variants import InsertionNotesV2RAG

    class AppRAG(InsertionNotesV2RAG):
        cache_path = Path("logs/app_rewrite_cache.json")   # keep visitors' questions out of eval/

    return AppRAG()


def render(r):
    st.markdown(r.display())
    with st.expander("How this answer was found"):
        st.write("Articles retrieved:", ", ".join(g["key"] for g in r.retrieved) or "none")
        st.write("Articles cited:", ", ".join(r.cited_articles) or "none")
        st.caption(f"{r.timings.get('total_s', 0):.1f}s | "
                   f"{r.usage.get('total_tokens', 0)} tokens | model {r.model}")
        if r.error:
            st.error(r.error)


st.title("⚖️ Law_bot")
st.caption("Answers come only from the text of the Constitution of Pakistan "
           "(as amended up to the 27th Amendment, 2025). Not legal advice.")

if "history" not in st.session_state:
    st.session_state.history, st.session_state.count = [], 0

with st.sidebar:
    st.subheader("Try one")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.caption(f"Questions this session: {st.session_state.count}/{MAX_QUESTIONS_PER_SESSION}")

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
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the Constitution..."):
                r = load_rag().ask(question[:500])
            render(r)
        st.session_state.history.append((question, r))
        st.session_state.count += 1