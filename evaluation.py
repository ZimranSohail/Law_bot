import json
from pathlib import Path
import re
from config import settings, client
EVAL_PATH = Path("eval/questions.json")


def load_eval(path=EVAL_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_retrieval(rag_obj, items, ks=(1, 3, 5), verbose=True):
    rows = []
    for it in items:
        if not it["expected_articles"]:
            continue                                  # out-of-scope: judged on the answer, not retrieval
        q = it["question"]
        groups, _ = rag_obj.retrieve(q, rag_obj.detect(q))
        keys = [g.key for g in groups]
        ranks = [keys.index(a) + 1 for a in it["expected_articles"] if a in keys]
        rows.append({"id": it["id"], "category": it["category"], "keys": keys,
                     "first_rank": min(ranks) if ranks else None,
                     "recall": len(ranks) / len(it["expected_articles"])})
    n = len(rows)
    summary = {f"hit@{k}": sum(1 for r in rows if r["first_rank"] and r["first_rank"] <= k) / n for k in ks}
    summary["mean_recall"] = sum(r["recall"] for r in rows) / n
    if verbose:
        for r in rows:
            mark = "MISS" if r["first_rank"] is None else f"rank {r['first_rank']}"
            print(f"{r['id']} {r['category']:14} {mark:7} recall={r['recall']:.1f} top={r['keys'][:5]}")
        print(f"\n{rag_obj.name}: " + " | ".join(f"{k}={v:.2f}" for k, v in summary.items()) + f"  (n={n})")
    return rows, summary


JUDGE_PROMPT = """You are grading an answer about the Constitution of Pakistan.
For EACH expected fact, decide:
- "supported": the answer EXPLICITLY states the fact (a paraphrase counts).
- "missing": the answer does not mention it. Absence is NEVER "supported".
- "contradicted": the answer states something incompatible with it.
Return ONLY JSON: {"facts": [{"fact": "...", "verdict": "supported" | "missing" | "contradicted"}]}"""

DECLINE_PROMPT = """Decide whether the answer DECLINES to answer because the provided sources / the Constitution do not cover the question, WITHOUT stating an answer from outside knowledge.
Return ONLY JSON: {"declines": true | false, "reason": "<one short sentence>"}"""


def _judge(system_prompt, user_text):
    kwargs = dict(model=settings.deployment,
                  messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}],
                  max_completion_tokens=600)
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    text = (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


def score_answers(system, items):
    rows = []
    for it in items:
        facts, decline = it.get("expected_facts"), it.get("should_decline")
        if not facts and not decline:
            continue
        r = system.ask(it["question"])
        row = {"id": it["id"], "answer": r.answer, "error": r.error, "decline_expected": bool(decline)}
        try:
            if decline:
                j = _judge(DECLINE_PROMPT, f"Question: {it['question']}\n\nAnswer:\n{r.answer}")
                row["declines"], row["note"] = bool(j["declines"]), j.get("reason", "")
            else:
                j = _judge(JUDGE_PROMPT, f"Question: {it['question']}\n\nAnswer:\n{r.answer}\n\n"
                                         f"Expected facts:\n{json.dumps(facts)}")
                row["verdicts"] = [f["verdict"] for f in j["facts"]]
                if len(row["verdicts"]) != len(facts):
                    row["judge_error"] = "fact count mismatch"
        except Exception as e:
            row["judge_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        rows.append(row)
    return rows


def summarize(name, rows):
    fr = [r for r in rows if "verdicts" in r and "judge_error" not in r]
    v = [x for r in fr for x in r["verdicts"]]
    dr = [r for r in rows if r["decline_expected"] and "declines" in r]
    errs = sum(1 for r in rows if "judge_error" in r or r["error"])
    print(f"{name:20}: facts supported {v.count('supported')}/{len(v)} | "
          f"contradicted {v.count('contradicted')} | "
          f"declines correctly {sum(r['declines'] for r in dr)}/{len(dr)} | errors {errs}")