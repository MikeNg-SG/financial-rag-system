import re
import json
from pathlib import Path

# Split the clean and reabale text into each item
def split_by_item(clean_text: str) -> dict[str, str]:
    # Find every Item heading (e.g. "Item 7.") and its position in the text, regardless of upper/lower case
    pattern = r"\n(Item\s+\d+[A-C]?\.)"
    matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))

    items = {}
    for i, match in enumerate(matches):
        item_label = match.group(1).strip().title()
        start = match.end()  # text starts right after this Item's heading
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
        section_text = clean_text[start:end].strip()

        items[item_label] = section_text

    return items


# Keep specific items for RAG system
def filter_target_items(items: dict[str, str], target_labels: list[str]) -> dict[str, str]:
    filtered = {}
    for label in target_labels:
        if label in items:
            filtered[label] = items[label]
        else:
            print(f"Warning: {label} not found in this document")

    return filtered

def section_all_companies(companies: list[str], target_label: list[str], input_dir = 'data/processed', output_dir = 'data/processed') -> dict[str, dict[str,str]]:
    all_sections = {}
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    for ticker in companies:
        clean_file = in_path / f"{ticker}_10k_clean.txt"
        clean_text = clean_file.read_text(encoding="utf-8")

        print(f"Sectioning {ticker}...")
        items = split_by_item(clean_text)
        filtered = filter_target_items(items, target_labels)
        all_sections[ticker] = filtered

        output_file = out_path / f"{ticker}_sections.json"
        output_file.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
        print(f" -> saved {len(filtered)} sections to {output_file}\n")

    return all_sections

if __name__ == "__main__":
    companies = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    target_labels = ["Item 1.", "Item 1A.", "Item 7.", "Item 8."]

    all_sections = section_all_companies(companies, target_labels)
    print("Done. Sectioned companies:", list(all_sections.keys()))