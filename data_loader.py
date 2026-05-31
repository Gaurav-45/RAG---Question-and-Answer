from dotenv import load_dotenv
import os

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def load_and_chunk_pdf(file_path: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = splitter.split_documents(documents)

    return [chunk.page_content for chunk in chunks]


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )

    return [embedding.values for embedding in response.embeddings]