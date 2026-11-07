import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from config import settings
from evaluation import load_eval, score_retrieval, score_answers
from variants import (RewriteStrictRAG, RouterRAG, FusionRAG, InsertionNotesV2RAG,
                      ScopeGateRAG, CompleteRAG, BroadRAG)
from base_rag import BaseRAG

SYSTEMS = {
    "base": BaseRAG,
    "strict": RewriteStrictRAG,
    "router": RouterRAG,
    "fusion": FusionRAG,
    "fusion_insertion_v2": InsertionNotesV2RAG,
    "fusion_scope": ScopeGateRAG,
    "fusion_complete": CompleteRAG,
    "broad": BroadRAG,
}


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def passed(row):
    if "judge_error" in row or row.get("error"):
        return None          # infrastructure failure, not a verdict
    if row["decline_expected"]:
        return bool(row.get("declines"))
    v = row.get("verdicts", [])
    return bool(v) and all(x == "supported" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="base", choices=list(SYSTEMS))
    ap.add_argument("--split", nargs="*", default=None,
                    help="e.g. dev heldout heldout2 heldout3 (default: all)")
    ap.add_argument("--answers", action="store_true", help="also score answers (LLM calls)")
    ap.add_argument("--runs", type=int, default=3, help="answer runs per question")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--db-path", default=None, help="Chroma folder (default: VectorStore's default)")
    args = ap.parse_args()

    items = load_eval()
    if args.split:
        items = [it for it in items if it.get("split", "dev") in args.split]
    if args.verified_only:
        items = [it for it in items if it.get("verified")]

    kw = {"log_path": "logs/eval_runs.jsonl"}
    if args.db_path:
        from store import VectorStore
        kw["store"] = VectorStore(path=args.db_path)
    system = SYSTEMS[args.system](**kw)

    report = {
        "system": args.system, "model": settings.deployment,
        "temperature": settings.temperature, "git": git_commit(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": args.split or "all", "n_questions": len(items),
        "db_path": args.db_path or "default",
    }

    # ---- retrieval (no LLM calls beyond cached rewrites) ----
    rows, summary = score_retrieval(system, items, verbose=False)
    n = len(rows)
    ret = {"n": n}
    for k in (1, 3, 5):
        hits = round(summary[f"hit@{k}"] * n)
        lo, hi = wilson(hits, n)
        ret[f"hit@{k}"] = {"value": round(summary[f"hit@{k}"], 3), "ci95": [round(lo, 2), round(hi, 2)]}
    ret["mean_recall"] = round(summary["mean_recall"], 3)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    ret["by_category_hit@5"] = {c: f"{sum(1 for r in rs if r['first_rank'] and r['first_rank'] <= 5)}/{len(rs)}"
                                for c, rs in sorted(by_cat.items())}
    ret["misses"] = [r["id"] for r in rows if r["first_rank"] is None]
    report["retrieval"] = ret

    print(f"\n{args.system} | {settings.deployment} | {len(items)} questions")
    print(f"retrieval (n={n}): " + " | ".join(
        f"hit@{k}={ret[f'hit@{k}']['value']:.2f} CI{ret[f'hit@{k}']['ci95']}" for k in (1, 3, 5)))
    print("  by category (hit@5):", ret["by_category_hit@5"])
    print("  misses:", ret["misses"])

    # ---- answers (LLM calls: ~ runs x questions asks + judge calls) ----
    if args.answers:
        a_items = [it for it in items if it.get("expected_facts") or it.get("should_decline")]
        per_q = defaultdict(list)
        samples = {}
        errors = 0
        for _ in range(args.runs):
            for row in score_answers(system, a_items):
                p = passed(row)
                if p is None:
                    errors += 1
                    continue
                per_q[row["id"]].append(p)
                if not p:
                    samples[row["id"]] = row["answer"][:500]
        cat_of = {it["id"]: it["category"] for it in a_items}
        majority = [qid for qid, p in per_q.items() if sum(p) / len(p) >= 0.5]
        lo, hi = wilson(len(majority), len(per_q))
        mean_rate = sum(sum(p) / len(p) for p in per_q.values()) / max(len(per_q), 1)
        cat = defaultdict(list)
        for qid, p in per_q.items():
            cat[cat_of[qid]].append(sum(p) / len(p))
        report["answers"] = {
            "n_questions": len(per_q), "runs": args.runs,
            "mean_pass_rate": round(mean_rate, 3),
            "questions_passing_majority": f"{len(majority)}/{len(per_q)}",
            "ci95_majority": [round(lo, 2), round(hi, 2)],
            "by_category": {c: round(sum(v) / len(v), 2) for c, v in sorted(cat.items())},
            "per_question": {q: f"{sum(p)}/{len(p)}" for q, p in sorted(per_q.items())},
            "failing_examples": samples,
            "errored_runs": errors,
        }
        print(f"\nanswers (n={len(per_q)} questions x {args.runs} runs): "
              f"mean pass {mean_rate:.2f} | majority-pass {len(majority)}/{len(per_q)} "
              f"CI{report['answers']['ci95_majority']}")
        print(f"  errored runs (excluded): {errors}")
        print("  failing:", {q: v for q, v in report["answers"]["per_question"].items()
                             if q in [x for x, p in per_q.items() if not all(p)]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.system}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved", path)


if __name__ == "__main__":
    main()