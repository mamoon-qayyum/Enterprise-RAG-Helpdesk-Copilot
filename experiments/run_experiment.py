from pathlib import Path
import argparse
import csv
import json

from src.research_pipeline import ResearchRAG

BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config JSON, e.g. configs/c1.json")
    p.add_argument("--split", choices=["dev", "test"], required=True)
    return p.parse_args()


def load_benchmark(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    config_path = BASE_DIR / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    benchmark_path = BASE_DIR / "benchmark" / f"{args.split}.csv"
    rows = load_benchmark(benchmark_path)
    if not rows:
        raise ValueError(f"Benchmark is empty: {benchmark_path}")

    rag = ResearchRAG()
    output = []

    for i, row in enumerate(rows, start=1):
        qid = row["id"].strip()
        question = row["question"].strip()
        print(f"[{i}/{len(rows)}] {cfg['name']} {args.split} {qid}: {question}")

        result = rag.answer(
            question=question,
            threshold=cfg.get("threshold"),
            grounding=cfg.get("grounding", "strong"),
            k=int(cfg.get("k", 3)),
            use_sufficiency_check=bool(cfg.get("use_sufficiency_check", False)),
        )

        answerable = row["answerable"].strip() == "1"
        expected_source = row.get("expected_source", "").strip()
        expected_keywords = [
            x.strip().lower() for x in row.get("expected_keywords", "").split(";") if x.strip()
        ]
        answer_lower = result["answer"].lower()

        retrieval_pass = bool(expected_source) and expected_source in result["sources"]
        keyword_pass = bool(expected_keywords) and all(k in answer_lower for k in expected_keywords)

        output.append({
            "id": qid,
            "split": args.split,
            "category": row.get("category", ""),
            "question": question,
            "ground_truth_answerable": int(answerable),
            "expected_source": expected_source,
            "expected_keywords": ";".join(expected_keywords),
            "configuration": cfg["name"],
            "threshold": "" if cfg.get("threshold") is None else cfg.get("threshold"),
            "grounding": cfg.get("grounding", ""),
            "k": cfg.get("k", 3),
            "use_sufficiency_check": int(bool(cfg.get("use_sufficiency_check", False))),
            "best_score": "" if result["best_score"] is None else f"{result['best_score']:.6f}",
            "all_scores": ";".join(f"{x:.6f}" for x in result["scores"]),
            "retrieved_sources": ";".join(result["sources"]),
            "retrieval_pass": int(retrieval_pass),
            "system_decision": result["decision"],
            "sufficiency_label": result["sufficiency_label"],
            "keyword_answer_pass": int(keyword_pass),
            "answer": result["answer"],
        })

    out_dir = BASE_DIR / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['name'].lower()}_{args.split}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    print(f"Saved {len(output)} rows to {out_path}")


if __name__ == "__main__":
    main()
