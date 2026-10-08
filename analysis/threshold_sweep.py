from pathlib import Path
import argparse
import csv


def div(a,b): return a/b if b else 0.0

p = argparse.ArgumentParser()
p.add_argument("results_csv", help="Any raw results CSV containing best_score and ground truth")
p.add_argument("--min", dest="min_v", type=float, default=0.0)
p.add_argument("--max", dest="max_v", type=float, default=0.5)
p.add_argument("--step", type=float, default=0.025)
args = p.parse_args()

with open(args.results_csv, "r", encoding="utf-8-sig", newline="") as f:
    rows=list(csv.DictReader(f))

thresholds=[]
x=args.min_v
while x <= args.max_v + 1e-12:
    thresholds.append(round(x,6)); x += args.step

out=[]
for t in thresholds:
    tp=fp=tn=fn=0
    for r in rows:
        if r.get("best_score","") == "":
            continue
        gt = r["ground_truth_answerable"] == "1"
        predicted_answerable = float(r["best_score"]) >= t
        if gt and predicted_answerable: tp += 1
        elif (not gt) and predicted_answerable: fp += 1
        elif (not gt) and (not predicted_answerable): tn += 1
        else: fn += 1
    p_=div(tp,tp+fp); rec=div(tp,tp+fn); f1=div(2*p_*rec,p_+rec)
    out.append({
        "threshold": t,
        "precision": round(p_,4),
        "recall": round(rec,4),
        "f1": round(f1,4),
        "false_refusal_rate": round(div(fn,tp+fn),4),
        "unsupported_answer_rate": round(div(fp,fp+tn),4),
    })

out_path=Path("results/summary/threshold_sweep.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path,"w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print(f"Saved {out_path}")
