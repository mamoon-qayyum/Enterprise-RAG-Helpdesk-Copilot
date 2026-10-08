# Paper-answerability starter

This starter is designed to be copied into the root of your existing `Enterprise-RAG-Helpdesk-Copilot` repository on a new branch named `paper-answerability`.

## 1. Create the branch locally

```bash
git switch main
git pull
git switch -c paper-answerability
```

## 2. Copy the starter contents into the repository root
Merge the folders from this starter with your existing repository. Do not delete `app.py`, `src/ingest.py`, `src/rag_chat.py`, `src/retrieve.py`, or `data/`.

## 3. Preserve the MSc evaluation

```bash
bash scripts/prepare_branch.sh
```

## 4. Install the paper-only dependency

```bash
pip install -r requirements.txt
pip install -r requirements-paper.txt
```

Your existing code imports `python-docx` and `openpyxl` in `src/ingest.py`; if they are not already installed in your environment, add/install them before rebuilding the database.

## 5. Build/fill benchmark
- Use `benchmark/dev.csv` for prompt/configuration tuning.
- Keep `benchmark/test.csv` sealed until C1-C5 are frozen.

## 6. Run experiments

```bash
python experiments/run_experiment.py --config configs/c1.json --split dev
python experiments/run_experiment.py --config configs/c2.json --split dev
python experiments/run_experiment.py --config configs/c3.json --split dev
python experiments/run_experiment.py --config configs/c4.json --split dev
python experiments/run_experiment.py --config configs/c5.json --split dev
```

Once the design is frozen, run the same five commands with `--split test`.

## 7. Summarize

```bash
python analysis/summarise_all.py
python evaluation/metrics.py results/raw/c5_test.csv
python analysis/threshold_sweep.py results/raw/c5_test.csv
python analysis/plot_results.py
```

## 8. Commit

```bash
git add .
git commit -m "Add answerability and abstention research experiment framework"
git push -u origin paper-answerability
```

Important: the explicit sufficiency check in C5 adds an extra LLM call per query. Record this as a cost/latency limitation in the paper.
