import argparse
import sys

from variants import InsertionNotesV2RAG, ScopeGateRAG, FusionRAG
from base_rag import BaseRAG
from pathlib import Path

def with_private_cache(cls):
    return type(cls.__name__, (cls,), {"cache_path": Path("logs/ask_rewrite_cache.json")})

SYSTEMS = {
    "base": BaseRAG,
    "fusion": with_private_cache(FusionRAG),
    "default": with_private_cache(InsertionNotesV2RAG),
    "scope": with_private_cache(ScopeGateRAG),
}


def main():
    ap = argparse.ArgumentParser(description="Ask the Constitution of Pakistan bot")
    ap.add_argument("question", nargs="*", help="question (omit for interactive mode)")
    ap.add_argument("--system", default="default", choices=list(SYSTEMS))
    ap.add_argument("--trace", action="store_true", help="show retrieved articles and timings")
    args = ap.parse_args()

    rag = SYSTEMS[args.system]()

    def run(q):
        r = rag.ask(q)
        print("\n" + r.display())
        if args.trace:
            print("\n-- trace --")
            print("detected:", r.detected_articles, "| missing:", r.missing_articles)
            print("retrieved:", [(g["key"], g["via"]) for g in r.retrieved])
            print("cited:", r.cited_articles, "| not retrieved:", r.cited_not_retrieved)
            print("timings:", r.timings, "| tokens:", r.usage.get("total_tokens"))
            if r.error:
                print("ERROR:", r.error)
        print()

    if args.question:
        run(" ".join(args.question))
        return
    print("Ask about the Constitution of Pakistan (Ctrl+C or empty line to quit)")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    sys.exit(main())