from typing import List, Optional
from langchain.schema import Document

from src.ingestion.document_loader import create_document_loader
from src.ingestion.text_splitter import create_text_splitter
from src.ingestion.embedding_generator import create_embedding_generator
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import Retriever
from src.generation.llm_client import create_llm_client
from src.generation.prompt_templates import RAGPromptTemplate

class RAGPipeline:
    """Main RAG pipeline orchestrating the entire flow"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        # Embedding generator
        self.embeddings = create_embedding_generator(
            provider=self.config.get("embedding_provider", "openai"),
            model=self.config.get("embedding_model", "text-embedding-3-large")
        )
        
        # Vector store
        self.vector_store = create_vector_store(
            collection_name=self.config.get("collection_name", "documents"),
            embedding_function=self.embeddings.embeddings,
            persist_directory=self.config.get("persist_directory", "./chroma_db")
        )
        
        # Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            reranker=None  # Add reranker if needed
        )
        
        # Text splitter
        self.text_splitter = create_text_splitter(
            splitter_type=self.config.get("splitter_type", "recursive"),
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        
        # LLM client
        self.llm_client = create_llm_client(
            provider=self.config.get("llm_provider", "deepseek"),
            model=self.config.get("llm_model", "deepseek-chat"),
            temperature=self.config.get("temperature", 0.1)
        )
        
        # Prompt template
        self.prompt_template = RAGPromptTemplate()
    
    def ingest_document(self, file_path: str) -> None:
        """Ingest a document into the vector store"""
        try:
            # Load document
            loader = create_document_loader(file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            print(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to ingest document {file_path}: {e}")
    
    def query(self, question: str, k: int = 5) -> dict:
        """Query the RAG system"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(question, k=k)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Format context
            context = self._format_context(relevant_docs)
            
            # Generate prompt
            prompt = self.prompt_template.create_prompt(question, context)
            
            # Generate response
            answer = self.llm_client.generate_text(prompt)
            
            # Extract sources
            sources = self._extract_sources(relevant_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": self._calculate_confidence(relevant_docs)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process query: {e}")
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content = doc.page_content.strip()
            
            context_parts.append(f"[{i}] Source: {source}, Page: {page}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[dict]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "title": doc.metadata.get("title", "")
            })
        return sources
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents"""
        # Simple confidence calculation - can be improved
        if not documents:
            return 0.0
        
        # Higher confidence if more documents found
        confidence = min(len(documents) / 5.0, 1.0)
        return round(confidence, 2)
    
    def clear_vector_store(self) -> None:
        """Clear all documents from vector store"""
        try:
            self.vector_store.delete_collection()
            print("Vector store cleared successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clear vector store: {e}")
        
class RAGPipelineBuilder:
    """Builder class for RAG pipeline configuration"""
    
    def __init__(self, embedding_provider: str = "openai",
                 embedding_model: str = "text-embedding-3-large",
                    llm_provider: str = "deepseek",
                    llm_model: str = "deepseek-chat",
                    collection_name: str = "documents",
                    persist_directory: str = "./chroma_db",
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    temperature: float = 0.1):
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
    
    def set_embedding_provider(self, embedding_provider: str) -> 'RAGPipelineBuilder':
        self.embedding_provider = embedding_provider
        return self
    
    def set_embedding_model(self, model: str) -> 'RAGPipelineBuilder':
        self.embedding_model = model
        return self
    
    def set_llm_provider(self, llm_provider: str) -> 'RAGPipelineBuilder':
        self.llm_provider = llm_provider
        return self
    
    def set_llm_model(self, model: str) -> 'RAGPipelineBuilder':
        self.llm_model = model
        return self
    
    def set_collection_name(self, collection_name: str) -> 'RAGPipelineBuilder':
        self.collection_name = collection_name
        return self
    
    def set_persist_directory(self, persit_directory: str) -> 'RAGPipelineBuilder':
        self.persist_directory = persit_directory
        return self
    
    def set_chunk_size(self, chunk_size: int) -> 'RAGPipelineBuilder':
        self.chunk_size = chunk_size
        return self
    
    def set_chunk_overlap(self, chunk_overlap: int) -> 'RAGPipelineBuilder':
        self.chunk_overlap = chunk_overlap
        return self
    
    def set_temperature(self, temperature: float) -> 'RAGPipelineBuilder':
        self.temperature = temperature
        return self
    
    def build(self) -> RAGPipeline:
        """Build the RAG pipeline with the configured settings"""
        return self

# Configuration helper
def create_rag_pipeline(config_path: str = None) -> RAGPipeline:
    """Factory function to create RAG pipeline with configuration"""
    if config_path:
       
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "collection_name": "documents",
            "persist_directory": "./chroma_db",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "temperature": 0.1
        }
    
    return RAGPipeline(config)
