from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
from 

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

class MongoDBVectorStore(VectorStoreInterface):
    """MongoDB vector store implementation"""
    
    def __init__(self, 
                 collection_name: str,
                 embedding_function,
                 persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        # MongoDB initialization logic would go here
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the MongoDB vector store"""
        pass  # Implement MongoDB logic here
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search in MongoDB"""
        pass  # Implement MongoDB logic here
    
    def delete_collection(self) -> None:
        """Delete the MongoDB collection"""
        pass  # Implement MongoDB logic here

class VectorStore:

    def __init__(self, vector_store: VectorStoreInterface):
        self._vector_store = vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self._vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        return self._vector_store.similarity_search(query, k)
    
    def delete_collection(self) -> None:
        """Delete the vector store collection"""
        self._vector_store.delete_collection()

    def set_vector_store(self, vector_store: VectorStoreInterface):
        """Set a new vector store implementation"""
        self._vector_store = vector_store
    
# Factory function for easy instantiation
def create_vector_store(
    collection_name: str,
    embedding_function,
    persist_directory: Optional[str] = None,
    vector_store_type: str = "chroma"
) -> VectorStoreInterface:
    """Factory function to create a vector store instance"""
    if vector_store_type.lower() == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    elif vector_store_type.lower() == "mongodb":
        return MongoDBVectorStore(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")