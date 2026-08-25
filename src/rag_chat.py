from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


# -------------------------
# 1. Project paths
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# -------------------------
# 2. Load environment variables
# -------------------------

load_dotenv(BASE_DIR / ".env")


# -------------------------
# 3. Load embedding model
# -------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# 4. Connect to ChromaDB
# -------------------------

vector_store = Chroma(
    collection_name="hatfield_helpdesk",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


# -------------------------
# 5. Load the language model
# -------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# -------------------------
# 6. Ask the user a question
# -------------------------

question = input("\nAsk a question: ")


# -------------------------
# 7. Retrieve relevant documents
# -------------------------

results = vector_store.similarity_search(
    question,
    k=3
)


# -------------------------
# 8. Combine retrieved chunks
# -------------------------

context_parts = []

for document in results:
    source = document.metadata.get("source", "Unknown source")

    context_parts.append(
        f"Source: {source}\n"
        f"{document.page_content}"
    )

context = "\n\n".join(context_parts)


# -------------------------
# 9. Create RAG prompt
# -------------------------

prompt = f"""
You are the internal helpdesk assistant for Hatfield Technologies Ltd.

Answer the employee's question using ONLY the information contained
in the provided company documents.

Rules:
- Do not invent company policies.
- If the answer cannot be found in the provided context, say:
  "I could not find this information in the available company documents."
- Keep the answer clear and concise.
- Mention the source document(s) used.

Context:
{context}

Employee question:
{question}

Answer:
"""


# -------------------------
# 10. Ask the LLM
# -------------------------

response = llm.invoke(prompt)


# -------------------------
# 11. Display answer
# -------------------------

print("\nAnswer:")
print(response.content)