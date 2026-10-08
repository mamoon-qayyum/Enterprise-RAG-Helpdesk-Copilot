from pathlib import Path
import argparse
import csv


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_csv")
    args = p.parse_args()

    path = Path(args.results_csv)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    tp = fp = tn = fn = 0
    retrievable = retrieval_hits = 0
    keyword_total = keyword_hits = 0

    for r in rows:
        gt = r["ground_truth_answerable"] == "1"
        answered = r["system_decision"] == "answer"

        if gt and answered: tp += 1
        elif not gt and answered: fp += 1
        elif not gt and not answered: tn += 1
        elif gt and not answered: fn += 1

        if gt and r.get("expected_source", ""):
            retrievable += 1
            retrieval_hits += int(r.get("retrieval_pass", "0") or 0)

        if gt and r.get("expected_keywords", ""):
            keyword_total += 1
            keyword_hits += int(r.get("keyword_answer_pass", "0") or 0)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_refusal_rate = safe_div(fn, tp + fn)
    unsupported_answer_rate = safe_div(fp, fp + tn)
    retrieval_recall = safe_div(retrieval_hits, retrievable)
    keyword_accuracy = safe_div(keyword_hits, keyword_total)

    summary = {
        "file": str(path),
        "n": len(rows),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "answerability_precision": precision,
        "answerability_recall": recall,
        "answerability_f1": f1,
        "false_refusal_rate": false_refusal_rate,
        "unsupported_answer_rate": unsupported_answer_rate,
        "retrieval_recall_at_k": retrieval_recall,
        "keyword_answer_accuracy": keyword_accuracy,
    }

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
