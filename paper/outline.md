# Paper outline

## Abstract
Write last. Include problem, benchmark size, methods compared, main quantitative finding, and limitation.

## 1. Introduction
- Enterprise RAG must decide not only what is relevant, but whether evidence is sufficient.
- Fixed similarity thresholds may conflate semantic relevance with answerability.
- State RQ1-RQ4.
- Contributions: held-out benchmark; threshold/grounding ablation; explicit evidence-sufficiency comparison.

## 2. Related Work
- Retrieval-Augmented Generation
- Dense retrieval and semantic similarity
- Grounding / faithfulness
- RAG evaluation
- Abstention / selective answering

## 3. Methodology
- Existing enterprise corpus and ingestion
- Benchmark construction and dev/test split
- Experimental conditions C1-C5
- Metrics
- Reproducibility settings

## 4. Results
- Similarity distributions
- Threshold trade-off
- C1-C4 ablation
- C5 evidence-sufficiency comparison

## 5. Discussion
- Relevance vs evidential sufficiency
- Safety/utility trade-off
- Why configurations fail
- Cost/latency implications of an extra sufficiency call

## 6. Limitations
- Synthetic enterprise corpus
- Benchmark scale
- One embedding/generation model unless expanded
- Keyword check is diagnostic only
- No real enterprise users

## 7. Conclusion
Answer the research questions conservatively and avoid claims beyond the held-out benchmark.
