from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "results" / "raw"
OUT = BASE_DIR / "results" / "summary" / "main_results.csv"


def div(a,b): return a/b if b else 0.0

rows_out = []
for path in sorted(RAW_DIR.glob("*.csv")):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue

    tp=fp=tn=fn=0
    ret_n=ret_hit=kw_n=kw_hit=0
    for r in rows:
        gt = r["ground_truth_answerable"] == "1"
        ans = r["system_decision"] == "answer"
        if gt and ans: tp += 1
        elif (not gt) and ans: fp += 1
        elif (not gt) and (not ans): tn += 1
        else: fn += 1

        if gt and r.get("expected_source", ""):
            ret_n += 1; ret_hit += int(r.get("retrieval_pass", "0") or 0)
        if gt and r.get("expected_keywords", ""):
            kw_n += 1; kw_hit += int(r.get("keyword_answer_pass", "0") or 0)

    p=div(tp,tp+fp); rec=div(tp,tp+fn); f1=div(2*p*rec,p+rec)
    rows_out.append({
        "file": path.name,
        "configuration": rows[0]["configuration"],
        "split": rows[0]["split"],
        "n": len(rows),
        "answerability_precision": round(p,4),
        "answerability_recall": round(rec,4),
        "answerability_f1": round(f1,4),
        "false_refusal_rate": round(div(fn,tp+fn),4),
        "unsupported_answer_rate": round(div(fp,fp+tn),4),
        "retrieval_recall_at_k": round(div(ret_hit,ret_n),4),
        "keyword_answer_accuracy": round(div(kw_hit,kw_n),4),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
if rows_out:
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader(); w.writerows(rows_out)
    print(f"Saved {OUT}")
else:
    print("No raw result CSVs found.")
