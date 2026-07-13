import os

from src.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from a document. If the answer isn't in the context, "
    "say you don't know based on the given material. Be concise and cite "
    "which page(s) you drew the answer from."
)

# Free tier on Groq (console.groq.com). Good quality, fast, no cost.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for r in retrieved:
        parts.append(f"[Source: {r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retrieved: list[RetrievedChunk], model: str = DEFAULT_GROQ_MODEL) -> str:
    """Generate a grounded answer. Uses Groq if GROQ_API_KEY is set."""
    context_block = _build_context_block(retrieved)
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return _fallback_extractive_answer(retrieved)

    from groq import Groq
    client = Groq(api_key=api_key)

    user_prompt = (
        f"Context from document:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    completion = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def _fallback_extractive_answer(retrieved: list[RetrievedChunk]) -> str:
    """No-API-key fallback: just surface the best-matching passage."""
    if not retrieved:
        return "No relevant content was found in the document for this question."

    top = retrieved[0]
    note = (
        "(No GROQ_API_KEY set, so showing the most relevant passage found "
        "instead of a generated answer. Get a free key at console.groq.com "
        "to enable full generation.)"
    )
    return f"{note}\n\nFrom {top.chunk.source}, page {top.chunk.page}:\n\"{top.chunk.text}\""
