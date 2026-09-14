import streamlit as st

from financial_rag.vectorstore import connect_to_weaviate
from financial_rag.reranking import get_reranker, retrieve
from financial_rag.generation import build_prompt, generate_answer


st.set_page_config(page_title="Financial RAG", page_icon="📊", layout="centered")

# Connect to Weaviate
@st.cache_resource
def get_client():
    return connect_to_weaviate()

# Load the cross-encoder
@st.cache_resource
def get_reranker_model():
    return get_reranker()


st.title("📊 Financial RAG: Ask Big Tech's Annual Reports (FY2025)")
st.caption("Ask questions about Apple, Amazon, Google, and Microsoft's most recent annual reports (10-K filings)")

with st.expander("ℹ️ About this data"):
    st.markdown("""
    This app answers questions using the **most recently filed 10-K annual reports**
    (fiscal year 2025) for four companies:

    - **Apple (AAPL)** — fiscal year ended September 27, 2025
    - **Amazon (AMZN)** — fiscal year ended December 31, 2025
    - **Alphabet/Google (GOOGL)** — fiscal year ended December 31, 2025
    - **Microsoft (MSFT)** — fiscal year ended June 30, 2025

    All filings are sourced directly from the **SEC's EDGAR system**
    (sec.gov/edgar) — the official public database where U.S. companies are
    legally required to file financial disclosures. Answers are grounded in
    retrieved text from these filings, with sources shown below each answer.
    """)

st.write("**Try an example:**")
example_questions = [
    "What was Apple's total revenue?",
    "What did Amazon's management say about growth in AWS?",
    "What are the main risk factors Google identifies?",
    "What products and services does Microsoft offer?",
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

cols = st.columns(2)
for i, example in enumerate(example_questions):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.query_input = example

query = st.text_input("Your question:", key="query_input", placeholder="What was Apple's revenue?")

if st.button("Ask", type="primary") and query:
    try:
        client = get_client()
        reranker = get_reranker_model()

        with st.spinner("Retrieving relevant filings..."):
            top_chunks = retrieve(client, reranker, query, top_k=3)

        with st.spinner("Generating answer..."):
            prompt = build_prompt(query, top_chunks)
            answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, chunk in enumerate(top_chunks):
            with st.expander(f"Source {i+1}: {chunk['ticker']} — {chunk['item']}"):
                st.write(chunk["text"])

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info("Make sure Weaviate is running (docker ps) and your .env has a valid ANTHROPIC_API_KEY.")

st.divider()
st.caption("Built by Mike Nguyen · RAG pipeline over SEC 10-K filings · [GitHub](https://github.com/MikeNg-SG/financial-rag-system)")