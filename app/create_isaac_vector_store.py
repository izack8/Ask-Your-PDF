from src.pipeline.rag_pipeline import create_rag_pipeline


create_rag_pipeline(config_path="app/settings.json").ingest_document(file_path="app/isaactay_resume_2026.pdf")