import logging
from fastapi import FastAPI
import inngest
from inngest.experimental import ai
from google import genai
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunk=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunk
        source_id = chunk_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        db = QdrantStorage()
        db.upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunk_and_src = await ctx.step.run("load_and_chunk_pdf", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunk_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]

        db = QdrantStorage()
        search_result = db.search(query_vector, top_k)

        return RAGSearchResult(
            contexts=search_result["contexts"],
            sources=search_result["sources"]
        )

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    search_result = await ctx.step.run(
        "embed_and_search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    context_block = "\n\n".join(
        f"Context {i+1}:\n{context}"
        for i, context in enumerate(search_result.contexts)
    )

    prompt = f"""
        You are a helpful assistant.

        Answer the user's question using ONLY the provided context.
        If the answer cannot be found in the context, say:
        "I could not find that information in the provided documents."

        Context:
        {context_block}

        Question:
        {question}

        Answer:
    """

    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text.strip()

    return {
        "answer": answer,
        "sources": search_result.sources,
        "num_contexts": len(search_result.contexts)
    }

app= FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
#npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery