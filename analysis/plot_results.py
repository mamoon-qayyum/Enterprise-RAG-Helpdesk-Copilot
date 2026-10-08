from pathlib import Path
import csv
import matplotlib.pyplot as plt

summary = Path("results/summary/threshold_sweep.csv")
with open(summary,"r",encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))

x=[float(r["threshold"]) for r in rows]
false_refusal=[float(r["false_refusal_rate"]) for r in rows]
unsupported=[float(r["unsupported_answer_rate"]) for r in rows]

fig=plt.figure()
plt.plot(x,false_refusal,label="False refusal rate")
plt.plot(x,unsupported,label="Unsupported answer rate")
plt.xlabel("Similarity threshold")
plt.ylabel("Rate")
plt.title("Threshold trade-off")
plt.legend()
plt.tight_layout()
out=Path("results/figures/threshold_tradeoff.png")
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=200)
print(f"Saved {out}")
