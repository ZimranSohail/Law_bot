import json
from evaluation import load_eval, EVAL_PATH
from variants import InsertionNotesV2RAG, BroadRAG, BROAD_RE

items = load_eval()

# 1) false positives: which existing questions would the trigger catch?
hits = [(it["id"], it["question"]) for it in items
        if BROAD_RE.search(it["question"]) and "article" not in it["question"].lower()]
print("existing questions matching BROAD_RE:", hits or "none")

# 2) add fresh broad questions (retrieval-only; expected = articles that should be in the context)
NEW = [
    ("q50", "What are the rights of children?", ["25A", "11"]),
    ("q51", "What does the Constitution say about education?", ["25A", "37"]),
    ("q52", "What are the rights of an accused person?", ["10", "10A", "13"]),
    ("q53", "Tell me about freedom of expression and assembly", ["19", "16"]),
    ("q54", "What does the Constitution say about religious freedom?", ["20"]),
    ("q55", "what are the rights of minorities in pakistan", ["36", "20"]),     # seen before: dev
    ("q56", "tell me about the right of women", ["25", "34"]),                  # seen before: dev
]
have = {it["id"] for it in items}
for qid, q, exp in NEW:
    if qid not in have:
        items.append({"id": qid, "question": q, "expected_articles": exp, "category": "broad",
                      "should_decline": False, "verified": False,
                      "split": "dev" if qid in ("q55", "q56") else "heldout5"})
EVAL_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

base, broad = InsertionNotesV2RAG(log_path=None), BroadRAG(log_path=None)
store = base.store
print("\nverify the expected articles:")
for qid, q, exp in NEW:
    for a in exp:
        r = store.get_article(a)
        print(f"  {qid} {a}: {r['metadatas'][0].get('article_title') if r['ids'] else 'NOT FOUND'}")

# 3) recall and context size, default vs broad
print(f"\n{'':4}{'default recall':>16}{'broad recall':>14}{'chars default':>15}{'chars broad':>13}")
for it in [i for i in items if i["category"] == "broad"]:
    row = []
    for s in (base, broad):
        groups, _ = s.retrieve(it["question"], s.detect(it["question"]))
        keys = [g.key for g in groups]
        rec = sum(a in keys for a in it["expected_articles"]) / len(it["expected_articles"])
        ctx, _ = s.build_context(it["question"], groups)
        row += [rec, len(ctx)]
    print(f"{it['id']:4}{row[0]:>16.2f}{row[2]:>14.2f}{row[1]:>15}{row[3]:>13}")