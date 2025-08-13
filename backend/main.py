from src.pipeline.rag_pipline import create_rag_pipeline
import streamlit as st

def main():
    st.title("RAG Pipeline Demo")
    
    # Load configuration
    config = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
        "collection_name": "documents",
        "persist_directory": "./chroma_db",
        "splitter_type": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "llm_provider": "deepseek",
        "llm_model": "deepseek-chat",
        "temperature": 0.1
    }
    
    # Initialize RAG pipeline
    rag_pipeline = create_rag_pipeline()

    # Ingest document
    file_path = st.text_input("Enter the path of the document to ingest:")
    if st.button("Ingest Document"):
        rag_pipeline.ingest_document(file_path)
        st.success("Document ingested successfully!")

    # Query
    question = st.text_input("Enter your question:")
    if st.button("Ask Question"):
        response = rag_pipeline.query(question)
        st.write("Answer:", response.get("answer", ""))
        st.write("Sources:", response.get("sources", []))
        st.write("Confidence:", response.get("confidence", 0.0))

if __name__ == "__main__":
    main()