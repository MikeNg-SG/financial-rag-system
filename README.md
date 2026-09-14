# Financial RAG System

A RAG (Retrieval-Augmented Generation) system that ingests SEC 10-K filings 
from 4 companies (AAPL, AMZN, GOOGL, MSFT) and lets you ask questions about 
their financial reports, grounded in the actual filing text.

## What's been built so far

- **Ingestion**: Downloads the latest 10-K filing for each company directly 
  from SEC EDGAR's public API (`ingestion.py`)

- **Extraction**: Each SEC filing is a bundle of ~90 documents (the 10-K 
  itself, plus exhibits, XBRL data, images). Extraction pulls out just the 
  10-K document from that bundle (`extraction_10K.py`)

- **Cleaning**: Strips HTML/XBRL markup from the raw 10-K, leaving clean, 
  readable text. Also normalizes non-breaking spaces (`&nbsp;` → regular 
  space), since one company's filing used them inconsistently around 
  section headings (`cleaning.py`)

- **Sectioning**: Splits the cleaned text into labeled sections by SEC 
  "Item" number (Item 1, 1A, 7, 8), and keeps only the 4 sections relevant 
  to financial Q&A. Handles both uppercase and lowercase heading styles, 
  since different companies' filing agents format headings differently 
  (`sectioning.py`)

- **Chunking**: Splits each Item's text into ~200-word overlapping chunks, 
  tagged with metadata (company ticker, Item label, chunk index) so 
  retrieval later knows exactly where each piece of text came from 
  (`chunking.py`)

- **Embedding model**: Set up `sentence-transformers` (all-MiniLM-L6-v2) 
  to convert chunk text into 384-dimension vectors (`embedding.py`)

- **Vector database**: Set up Weaviate (running locally via Docker) with a 
  collection schema matching the chunk metadata, ready to store embedded 
  chunks (`vectorstore.py`)

## Real-world issues found and fixed along the way

- Different SEC filing agents (Workiva vs. DFIN) produce very different 
  HTML markup density for the same document type
- Google's 10-K uses uppercase "ITEM" headings, while Apple's uses 
  "Item" — required case-insensitive matching
- Amazon's filing mixes regular spaces and non-breaking spaces around 
  the same headings, causing silent data-quality bugs if not normalized

## Tech stack

- Python, `uv` for dependency/environment management
- `sec-edgar-downloader`, `beautifulsoup4` for data ingestion/cleaning
- `sentence-transformers` for embeddings
- Weaviate (via Docker) for vector storage

## Setup

1. `uv sync` — install dependencies
2. Start Weaviate locally:
