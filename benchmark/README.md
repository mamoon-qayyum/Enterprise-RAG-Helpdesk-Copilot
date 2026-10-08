# Benchmark instructions

Columns:
- `id`: unique identifier.
- `category`: direct, paraphrased, harder, unsupported, partially_supported, ambiguous.
- `question`: evaluation question.
- `answerable`: `1` if the corpus contains sufficient evidence, otherwise `0`.
- `expected_source`: source path/name for answerable questions; blank for unsupported questions.
- `expected_keywords`: semicolon-separated diagnostic keywords. These are only a lightweight correctness check, not a complete semantic evaluation.
- `notes`: benchmark-construction notes.

## Critical rule
Freeze prompts/configurations using `dev.csv`. Do not inspect model failures on `test.csv` until the system design is frozen.
