import os
from dotenv import load_dotenv
from anthropic import Anthropic

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = ""
    for chunk in chunks:
        context_block += f"[Source: {chunk['ticker']}, {chunk['item']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."
Always cite which company and Item the information came from.

Context:
{context_block}

Question: {query}

Answer:"""

    return prompt


load_dotenv()  # reads .env and makes ANTHROPIC_API_KEY available

client_anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Send the prompt to Claude and return the generated answer.
def generate_answer(prompt: str, model: str = "claude-haiku-4-5-20251001", temperature: float = 0.2, max_tokens: int = 500) -> str:
    response = client_anthropic.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text

if __name__ == "__main__":
    import weaviate
    from reranking import get_reranker, retrieve

    client = weaviate.connect_to_local()
    reranker = get_reranker()

    query = "What was Apple's revenue?"
    top_chunks = retrieve(client, reranker, query)

    prompt = build_prompt(query, top_chunks)
    answer = generate_answer(prompt)

    print("Question:", query)
    print("\nAnswer:")
    print(answer)

    client.close()