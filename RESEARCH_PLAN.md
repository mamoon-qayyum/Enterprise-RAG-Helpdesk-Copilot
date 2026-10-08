# Paper research plan

## Working title
Evaluating Answerability and Abstention Strategies in Enterprise Retrieval-Augmented Generation

## Research problem
Can an enterprise RAG system reliably determine whether retrieved evidence is sufficient to answer a question, and how does that compare with using a fixed vector-similarity threshold?

## Research questions
- RQ1: How well do vector-similarity scores distinguish answerable from unsupported enterprise questions?
- RQ2: How does similarity-threshold selection affect false refusals and unsupported answers?
- RQ3: Does stronger evidence-grounding improve answerability decisions independently of threshold selection?
- RQ4: Does an explicit evidence-sufficiency decision improve abstention compared with fixed similarity thresholds?

## Hypotheses
- H1: Similarity scores alone will not reliably separate answerable and unsupported queries.
- H2: Higher similarity thresholds will reduce unsupported answers but increase false refusals.
- H3: Stronger grounding instructions will reduce unsupported answers independently of threshold changes.
- H4: An explicit evidence-sufficiency decision will provide a better answer/refuse trade-off than a fixed similarity threshold.

## Experimental conditions
| ID | Threshold | Grounding | Explicit sufficiency check |
|---|---:|---|---|
| C1 | 0.30 | Original | No |
| C2 | 0.10 | Original | No |
| C3 | 0.30 | Strong | No |
| C4 | 0.10 | Strong | No |
| C5 | None | Strong | Yes |

## Benchmark target
- 50 development questions
- 100 held-out test questions
- Keep the held-out test set untouched until configurations and prompts are frozen.

Suggested categories across the full 150-question benchmark:
- 40 direct answerable
- 30 paraphrased answerable
- 20 harder/cross-document answerable
- 30 unsupported but topically related
- 15 partially supported
- 15 ambiguous

Do not include prompt-injection testing in this paper unless time remains after the core answerability study is complete.
