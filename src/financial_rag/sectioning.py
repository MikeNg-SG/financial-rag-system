import re

# Split the clean and reabale text into each item
def split_by_item(clean_text: str) -> dict[str, str]:
    pattern = r"\n(Item\s+\d+[A-C]?\.)"
    matches = list(re.finditer(pattern, clean_text))

    items = {}
    for i, match in enumerate(matches):
        item_label = match.group(1).strip()
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


if __name__ == "__main__":
    from cleaning import clean_html_to_text
    from extraction_10K import extract_10k_text

    filepath = "data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
    raw_html = extract_10k_text(filepath)
    clean_text = clean_html_to_text(raw_html)

    # List all items found
    all_items = split_by_item(clean_text)
    print(f"Found {len(all_items)} total sections\n")
    print(all_items.keys())

    # Filter selected items
    target_labels = ["Item 1.", "Item 1A.", "Item 7.", "Item 8."]
    filtered_items = filter_target_items(all_items, target_labels)

    print(f"\nKept {len(filtered_items)} target sections:")
    for label, text in filtered_items.items():
        print(f"  {label}: {len(text)} characters")