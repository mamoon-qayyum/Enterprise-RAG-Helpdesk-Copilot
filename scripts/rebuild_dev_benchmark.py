from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE = BASE_DIR / "results" / "raw" / "c1_dev.csv"
DESTINATION = BASE_DIR / "benchmark" / "dev.csv"

with open(SOURCE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

fieldnames = [
    "id",
    "category",
    "question",
    "answerable",
    "expected_source",
    "expected_keywords",
    "notes",
]

output = []

for row in rows:
    output.append({
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "answerable": row["ground_truth_answerable"],
        "expected_source": row["expected_source"],
        "expected_keywords": row["expected_keywords"],
        "notes": "Reconstructed from recorded development experiment",
    })

with open(DESTINATION, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output)

print(f"Restored {len(output)} development questions to {DESTINATION}")