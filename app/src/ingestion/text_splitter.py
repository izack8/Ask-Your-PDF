from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

class TextSplitterInterface(ABC):
    """Abstract base class for text splitters"""
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        pass

class RecursiveTextSplitter(TextSplitterInterface):
    """Recursive character text splitter implementation"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            # chunks = self._splitter.split_documents(documents)
            # for chunk in chunks:
            #     print("Chunk content:", repr(chunk.page_content))
            # return chunks
            return self._splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to split documents: {e}")

class TokenBasedSplitter(TextSplitterInterface):
    """Token-based text splitter implementation"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            return self._splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to split documents: {e}")

# Factory function
def create_text_splitter(splitter_type: str = "recursive", **kwargs) -> TextSplitterInterface:
    """Factory function to create text splitters"""
    if splitter_type.lower() == "recursive":
        return RecursiveTextSplitter(**kwargs)
    elif splitter_type.lower() == "token":
        return TokenBasedSplitter(**kwargs)
    else:
        raise ValueError(f"Unsupported splitter type: {splitter_type}")