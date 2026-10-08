from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
ABSTAIN_TEXT = "I could not find this information in the available company documents."

load_dotenv(BASE_DIR / ".env")

ORIGINAL_GROUNDING = """
You are the internal helpdesk assistant for Hatfield Technologies Ltd.

Answer the employee's question using ONLY the information contained in the provided company documents.

Rules:
- Do not invent company policies.
- If the answer cannot be found in the provided context, say exactly:
  \"I could not find this information in the available company documents.\"
- Keep the answer clear and concise.
""".strip()

STRONG_GROUNDING = """
You are the internal helpdesk assistant for Hatfield Technologies Ltd.

Answer using ONLY information explicitly supported by the retrieved company documents.

Rules:
- Do not use outside knowledge.
- Do not invent or infer company policies.
- First determine whether the retrieved evidence actually contains enough information to answer the question.
- Topic relevance is not sufficient: the requested claim must be explicitly supported.
- If evidence is missing, ambiguous, partial, or insufficient, respond exactly:
  \"I could not find this information in the available company documents.\"
- Keep supported answers clear and concise.
""".strip()

SUFFICIENCY_PROMPT = """
You are an evidence-sufficiency classifier for an enterprise RAG system.

Using ONLY the retrieved evidence below, decide whether the evidence is sufficient to answer the employee question completely and directly.

Return exactly one label and nothing else:
SUPPORTED
UNSUPPORTED
PARTIALLY_SUPPORTED

Definitions:
- SUPPORTED: the evidence explicitly contains enough information to answer the full question.
- UNSUPPORTED: the evidence does not contain the requested information.
- PARTIALLY_SUPPORTED: some, but not all, of the requested information is supported.

Retrieved evidence:
{context}

Employee question:
{question}

Label:
""".strip()


class ResearchRAG:
    def __init__(self, model="gpt-4.1-mini", embedding_model="text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            collection_name="hatfield_helpdesk",
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.llm = ChatOpenAI(model=model, temperature=0)

    @staticmethod
    def _make_context(results):
        parts = []
        sources = []
        scores = []
        for document, score in results:
            source = document.metadata.get("source", "Unknown source")
            parts.append(f"Source: {source}\n{document.page_content}")
            sources.append(source)
            scores.append(float(score))
        return "\n\n".join(parts), sources, scores

    def classify_sufficiency(self, question: str, context: str) -> str:
        prompt = SUFFICIENCY_PROMPT.format(context=context, question=question)
        label = self.llm.invoke(prompt).content.strip().upper()
        for candidate in ("PARTIALLY_SUPPORTED", "UNSUPPORTED", "SUPPORTED"):
            if candidate in label:
                return candidate
        return "UNSUPPORTED"

    def answer(self, question: str, threshold: Optional[float], grounding: str,
               k: int = 3, use_sufficiency_check: bool = False):
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=k)

        if not results:
            return {
                "answer": ABSTAIN_TEXT,
                "decision": "abstain",
                "sufficiency_label": "UNSUPPORTED",
                "sources": [],
                "scores": [],
                "best_score": None,
            }

        context, sources, scores = self._make_context(results)
        best_score = scores[0]

        # Threshold decision (C1-C4). Retrieval is still recorded even when rejected.
        if threshold is not None and best_score < threshold:
            return {
                "answer": ABSTAIN_TEXT,
                "decision": "abstain",
                "sufficiency_label": "NOT_RUN",
                "sources": sources,
                "scores": scores,
                "best_score": best_score,
            }

        sufficiency_label = "NOT_RUN"
        if use_sufficiency_check:
            sufficiency_label = self.classify_sufficiency(question, context)
            if sufficiency_label != "SUPPORTED":
                return {
                    "answer": ABSTAIN_TEXT,
                    "decision": "abstain",
                    "sufficiency_label": sufficiency_label,
                    "sources": sources,
                    "scores": scores,
                    "best_score": best_score,
                }

        grounding_prompt = ORIGINAL_GROUNDING if grounding == "original" else STRONG_GROUNDING
        prompt = f"""{grounding_prompt}

Retrieved company documents:
{context}

Employee question:
{question}

Answer:
"""
        answer = self.llm.invoke(prompt).content.strip()
        decision = "abstain" if ABSTAIN_TEXT.lower() in answer.lower() else "answer"

        return {
            "answer": answer,
            "decision": decision,
            "sufficiency_label": sufficiency_label,
            "sources": sources,
            "scores": scores,
            "best_score": best_score,
        }
