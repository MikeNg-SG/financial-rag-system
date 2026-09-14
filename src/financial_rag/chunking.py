import json
from pathlib import Path

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50)-> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def chunk_all_companies(companies: list[str], input_dir: str = "data/processed", output_dir: str = "data/processed") -> list[dict]:
    all_chunks = []
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for ticker in companies:
        sections_file = in_path / f"{ticker}_sections.json"
        sections = json.loads(sections_file.read_text(encoding="utf-8"))

        print(f"Chunking {ticker}...")
        company_chunks = []

        for item_label, item_text in sections.items():
            text_chunks = chunk_text(item_text)

            for i, chunk in enumerate(text_chunks):
                company_chunks.append({
                    "ticker": ticker,
                    "item": item_label,
                    "chunk_index": i,
                    "text": chunk,
                })

        all_chunks.extend(company_chunks)

        output_file = out_path / f"{ticker}_chunks.json"
        output_file.write_text(json.dumps(company_chunks, indent=2), encoding="utf-8")
        print(f"  → saved {len(company_chunks)} chunks to {output_file}\n")

    return all_chunks


if __name__ == "__main__":
    companies = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    all_chunks = chunk_all_companies(companies)
    print(f"Done. Total chunks across all companies: {len(all_chunks)}")
