from bs4 import BeautifulSoup
from pathlib import Path

# Remove HTML/XBRL tags, return a clean and readable plain text
def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    # Remove the hidden XBRL metadata block entirely 
    for hidden_div in soup.find_all(style=lambda s: s and "display:none" in s):
        hidden_div.decompose()

    text = soup.get_text(separator='\n')
    # Amazon uses non-breaking spaces (\xa0) in some headings — normalize them to regular spaces so Item labels match consistently
    text = text.replace('\xa0', ' ')
    return text

# Clean text for 4 companies
def clean_all_companies(companies: list[str], input_dir: str = 'data/processed', output_dir: str = 'data/processed') -> dict[str,str]:
    # Define Path
    cleaned = {}
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    # Start cleaning
    for ticker in companies:
        raw_file = in_path / f"{ticker}_10k_raw.txt"
        raw_html = raw_file.read_text(encoding="utf-8")

        print(f"Cleaning {ticker}...")
        clean_text = clean_html_to_text(raw_html)
        cleaned[ticker] = clean_text

        output_file = out_path / f"{ticker}_10k_clean.txt"
        output_file.write_text(clean_text, encoding="utf-8")
        print(f" -> saved to {output_file} ({len(clean_text)} characters)\n")

    return cleaned

if __name__ == "__main__":
    companies = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    cleaned = clean_all_companies(companies)
    print("Done. Cleaned companies:", list(cleaned.keys()))