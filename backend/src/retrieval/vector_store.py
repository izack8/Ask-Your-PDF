from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_chroma import Chroma
from langchain.schema import Document

class VectorStoreInterface(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        pass

class ChromaVectorStore(VectorStoreInterface):
    """Chroma vector store implementation"""
    
    def __init__(self, 
                 collection_name: str,
                 embedding_function,
                 persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self._store = None
        self._initialize_store()
    
    def _initialize_store(self) -> None:
        """Initialize the Chroma vector store"""
        try:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        except Exception as e:
            raise ConnectionError(f"failed to initialize Chroma store: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            self._store.add_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            return self._store.similarity_search(query, k=k)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")
    
    def delete_collection(self) -> None:
        """Delete the collection"""
        try:
            self._store.delete_collection()
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {e}")
    
    @property
    def store(self) -> Chroma:
        """Access to underlying Chroma store"""
        return self._store
    
# Factory function for easy instantiation
def create_vector_store(collection_name: str,
                        embedding_function,
                        persist_directory: str = "./chroma_db") -> VectorStoreInterface:
    """Factory function to create vector stores"""
    return ChromaVectorStore(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )