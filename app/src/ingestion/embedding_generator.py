from abc import ABC, abstractmethod
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class EmbeddingGeneratorInterface(ABC):
    """Abstract base class for embedding generators"""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        ...
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        ...

class OpenAIEmbeddingGenerator(EmbeddingGeneratorInterface):
    """OpenAI embedding generator implementation"""
    
    def __init__(self, 
                 model: str = "text-embedding-3-large",
                 api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI embeddings: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts for vector store"""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")
    
    @property
    def embeddings(self):
        """Get the underlying OpenAI embeddings object"""
        return self._embeddings
        
class EmbeddingGenerator:

    def __init__(self, embedding_generator: EmbeddingGeneratorInterface):
        self._embedding_generator = embedding_generator
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self._embedding_generator.generate_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self._embedding_generator.embed_query(text)
    
    def set_embedding_generator(self, embedding_generator: EmbeddingGeneratorInterface):
        """Set a new embedding generator"""
        self._embedding_generator = embedding_generator
    
    @property
    def embeddings(self) -> EmbeddingGeneratorInterface:
        """Access the underlying embedding generator"""
        return self._embedding_generator

# Factory function for easy instantiation
def create_embedding_generator(provider: str = "openai", **kwargs) -> EmbeddingGeneratorInterface:
    """Factory function to create embedding generators"""
    if provider.lower() == "openai":
        return EmbeddingGenerator(OpenAIEmbeddingGenerator(**kwargs))
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")