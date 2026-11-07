import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ALERT_ERROR_RATE = 0.05
ALERT_P95_SECONDS = 10.0
ALERT_UNRETRIEVED_CITATION_RATE = 0.10


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/rag_runs.jsonl")
    ap.add_argument("--last", type=int, default=None, help="only the most recent N runs")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M completion tokens")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"no log at {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.last:
        rows = rows[-args.last:]

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("variant"), r.get("model"))].append(r)

    alerts = []
    for (variant, model), rs in sorted(groups.items(), key=str):
        n = len(rs)
        errs = [r for r in rs if r.get("error")]
        ok = [r for r in rs if not r.get("error")]
        total = [r["timings"]["total_s"] for r in ok if r.get("timings", {}).get("total_s") is not None]
        gen = [r["timings"]["generate_s"] for r in ok if r.get("timings", {}).get("generate_s") is not None]
        ptok = [r["usage"].get("prompt_tokens", 0) for r in ok if r.get("usage")]
        ctok = [r["usage"].get("completion_tokens", 0) for r in ok if r.get("usage")]
        no_cite = sum(1 for r in ok if not r.get("cited_articles"))
        unret = sum(1 for r in ok if r.get("cited_not_retrieved"))
        dropped = sum(1 for r in ok if r.get("dropped_for_budget"))
        missing = sum(1 for r in ok if r.get("missing_articles"))

        print(f"\n=== {variant} | {model} | {n} runs ===")
        print(f"  errors            {len(errs)}/{n} ({len(errs)/n:.1%})")
        if total:
            print(f"  latency total     p50 {pct(total,50):.1f}s  p95 {pct(total,95):.1f}s  max {max(total):.1f}s")
        if gen:
            print(f"  latency generate  p50 {pct(gen,50):.1f}s  p95 {pct(gen,95):.1f}s")
        if ptok:
            print(f"  tokens/run        prompt avg {sum(ptok)/len(ptok):.0f}  completion avg {sum(ctok)/len(ctok):.0f}")
            if args.price_in is not None and args.price_out is not None:
                cost = sum(ptok) * args.price_in / 1e6 + sum(ctok) * args.price_out / 1e6
                print(f"  est. cost         ${cost:.4f} total, ${cost/len(ptok):.5f}/run")
        if ok:
            print(f"  no citations      {no_cite}/{len(ok)}  (declines or ungrounded answers)")
            print(f"  cited not retrieved {unret}/{len(ok)}  (rough flag: an article can be mentioned inside a source)")
            print(f"  context-budget drops {dropped}/{len(ok)} | asked for a missing article {missing}/{len(ok)}")
        if errs:
            kinds = Counter(r["error"].split(":")[0] for r in errs)
            print("  error types       ", dict(kinds))

        if len(errs) / n > ALERT_ERROR_RATE:
            alerts.append(f"{variant}: error rate {len(errs)/n:.1%} > {ALERT_ERROR_RATE:.0%}")
        if total and pct(total, 95) > ALERT_P95_SECONDS:
            alerts.append(f"{variant}: p95 latency {pct(total,95):.1f}s > {ALERT_P95_SECONDS}s")
        if ok and unret / len(ok) > ALERT_UNRETRIEVED_CITATION_RATE:
            alerts.append(f"{variant}: {unret/len(ok):.0%} of answers cite an unretrieved article")

    print("\nALERTS:" if alerts else "\nno alerts")
    for a in alerts:
        print("  -", a)


if __name__ == "__main__":
    main()