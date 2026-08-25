from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="hatfield_helpdesk",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


question = "How many annual leave days do employees get?"

results = vector_store.similarity_search(
    question,
    k=3
)


print(f"\nQuestion: {question}\n")

for i, document in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(f"Source: {document.metadata.get('source')}")
    print(document.page_content)
    print()