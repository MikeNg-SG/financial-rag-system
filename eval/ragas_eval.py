import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import embedding_factory
from ragas.metrics import Faithfulness, ResponseRelevancy

import weaviate
from financial_rag.vectorstore import connect_to_weaviate
from financial_rag.reranking import get_reranker, retrieve
from financial_rag.generation import build_prompt, generate_answer

# Loading the test questions
def load_test_questions(filepath='eval/test_queries.json') -> list[dict]:
    return json.loads(Path(filepath).read_text(encoding='utf-8'))

def build_ragas_dataset(client, reranker, test_questions: list[dict]) -> EvaluationDataset:
    """
    Run each test question through the full pipeline (retrieve + generate),
    and package the results into the format RAGAS expects.
    """
    samples = []

    for test in test_questions:
        question = test["question"]

        top_chunks = retrieve(client, reranker, question, top_k=3)
        contexts = [chunk["text"] for chunk in top_chunks]

        prompt = build_prompt(question, top_chunks)
        answer = generate_answer(prompt)

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        })

    return EvaluationDataset.from_list(samples)

from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

def get_judge():
    load_dotenv()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=2048)

    langchain_embeddings = LangChainOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    judge_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    return judge_llm, judge_embeddings

if __name__ == "__main__":
    client = connect_to_weaviate()
    reranker = get_reranker()

    test_questions = load_test_questions()

    print("Running pipeline on all test questions...")
    dataset = build_ragas_dataset(client, reranker, test_questions)

    judge_llm, judge_embeddings = get_judge()

    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== RAGAS RESULTS ===")
    print(results)

    df = results.to_pandas()
    print("\n--- Per-question breakdown ---")
    print(df[["user_input", "faithfulness", "answer_relevancy"]])
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ragas_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "timestamp": timestamp,
        "faithfulness_avg": float(df["faithfulness"].mean()),
        "answer_relevancy_avg": float(df["answer_relevancy"].mean()),
        "num_questions": len(df),
    }
    summary_path = output_dir / f"ragas_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults saved to {csv_path}")
    print(f"Summary saved to {summary_path}")

    client.close()