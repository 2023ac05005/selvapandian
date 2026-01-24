
import streamlit as st
from src.retrieve import HybridRetriever
from src.generator import Generator

st.set_page_config(page_title="Hybrid RAG (Dense+BM25+RRF)", layout="wide")
st.title("Hybrid RAG over Wikipedia (FAISS + BM25 + RRF)")

@st.cache_resource
def load_components():
    return HybridRetriever(), Generator()

retriever, generator = load_components()

q = st.text_input("Ask a question about the corpus:")
k_dense = st.slider("Dense top-K", 5, 50, 20)
k_sparse = st.slider("Sparse top-K", 5, 50, 20)
k_rrf = st.slider("RRF k", 10, 100, 60)
top_n = st.slider("Top-N fused", 2, 12, 8)

if st.button("Search & Answer") and q:
    with st.spinner("Retrieving..."):
        ctx = retriever.hybrid(q, k_dense, k_sparse, k_rrf, top_n)
    st.subheader("Answer")
    with st.spinner("Generating..."):
        ans = generator.generate(q, ctx)
    st.write(ans)

    st.subheader("Top contexts (with ranks)")
    for i, c in enumerate(ctx, 1):
        st.markdown(f"**[{i}] {c['title']}** — {c['url']}")
        st.write(f"Dense rank: {c['dense_rank']} · Sparse rank: {c['sparse_rank']}")
        with st.expander("Show text"):
            st.write(c["text"])
