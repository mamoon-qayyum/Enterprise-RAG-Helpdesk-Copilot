#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root AFTER creating the paper-answerability branch.
mkdir -p evaluation/original

if [ -f evaluation/evaluate.py ]; then
  cp evaluation/evaluate.py evaluation/original/msc_evaluate.py
fi
if [ -f evaluation/test_questions.json ]; then
  cp evaluation/test_questions.json evaluation/original/msc_test_questions.json
fi
if [ -f evaluation/results.csv ]; then
  cp evaluation/results.csv evaluation/original/msc_results.csv
fi

echo "Original MSc evaluation preserved under evaluation/original/."
echo "Now keep the old app/src files intact and use the new experiments/ pipeline for the paper."
