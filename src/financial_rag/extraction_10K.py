import re
from pathlib import Path


# Extract from 4 companies
def extract_10k_text(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)

    for doc in documents:
        type_match = re.search(r"<TYPE>(.*?)\n", doc)
        if type_match and type_match.group(1).strip() == "10-K":
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
            if text_match:
                return text_match.group(1)

    raise ValueError(f"No 10-K document found in {filepath}")

if __name__ == '__main__':
    companies = ['AAPL', 'AMZN', 'GOOGL', 'MSFT']
    # Create a dictionary to store the extracted text 
    extracted = {}
    # Create path to store the data
    output_dir = Path('data/processed')
    for ticker in companies:
        # Locate the company 10-K folder
        ticker_folder = Path(f'data/raw/sec-edgar-filings/{ticker}/10-K')

        subfolder = list(ticker_folder.iterdir())
        filling_folder = subfolder[0]
        filepath = filling_folder / 'full-submission.txt'
        print(f"Extracting {ticker}...")
        text = extract_10k_text(str(filepath))
        extracted[ticker] = text

        # Save the extracted text as its own clean file
        output_path = output_dir / f"{ticker}_10k_raw.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"  → saved to {output_path} ({len(text)} characters)\n")

    print("Done. Extracted companies:", list(extracted.keys()))





