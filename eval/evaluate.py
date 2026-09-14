import json
from pathlib import Path
from financial_rag.reranking import get_reranker, retrieve
from financial_rag.search import hybrid_search
import weaviate


def load_test_questions(filepath: str = "eval/test_queries.json") -> list[dict]:
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def check_retrieval(client, reranker, test_questions: list[dict]) -> list[dict]:
    results = []

    for test in test_questions:
        question = test["question"]
        expected_ticker = test["expected_ticker"]

        top_chunks = retrieve(client, reranker, question, top_k=3)

        retrieved_tickers = [chunk["ticker"] for chunk in top_chunks]

        if expected_ticker is None:
            passed = None
        else:
            passed = expected_ticker in retrieved_tickers

        results.append({
            "question": question,
            "expected_ticker": expected_ticker,
            "retrieved_tickers": retrieved_tickers,
            "passed": passed,
        })

    return results


if __name__ == "__main__":
    client = weaviate.connect_to_local()
    reranker = get_reranker()

    test_questions = load_test_questions()
    results = check_retrieval(client, reranker, test_questions)

    print("=== RETRIEVAL ACCURACY ===\n")
    correct = 0
    total_checkable = 0

    for r in results:
        status = "PASS" if r["passed"] else ("N/A" if r["passed"] is None else "FAIL")
        print(f"{status} | {r['question']}")
        print(f"   Expected: {r['expected_ticker']} | Got: {r['retrieved_tickers']}")

        if r["passed"] is not None:
            total_checkable += 1
            if r["passed"]:
                correct += 1

    print(f"\nRetrieval accuracy: {correct}/{total_checkable} ({correct/total_checkable*100:.0f}%)")

    client.close()