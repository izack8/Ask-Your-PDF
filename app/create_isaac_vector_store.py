from src.pipeline.rag_pipeline import create_rag_pipeline
create_rag_pipeline(config_path="settings.json").ingest_document(file_path="isaactay_resume_2025.pdf")