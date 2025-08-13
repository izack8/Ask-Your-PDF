from abc import ABC, abstractmethod
from typing import List
import fitz  # PyMuPDF - add this import
from langchain.schema import Document

class DocumentLoaderInterface(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        pass

class PDFLoader(DocumentLoaderInterface):
    def __init__(self, file_path: str, extract_metadata: bool = True):
        self.file_path = file_path
        self.extract_metadata = extract_metadata
    
    def load(self) -> List[Document]:
        try:
            doc = fitz.open(self.file_path)
            documents = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    continue
                    
                metadata = {
                    "source": self.file_path,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
                
                # Extract document metadata from first page
                if self.extract_metadata and page_num == 0:
                    doc_metadata = doc.metadata
                    if doc_metadata:
                        metadata.update({
                            "title": doc_metadata.get("title", ""),
                            "author": doc_metadata.get("author", ""),
                            "subject": doc_metadata.get("subject", ""),
                            "creator": doc_metadata.get("creator", "")
                        })
                
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
            
            doc.close()
            return documents
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF {self.file_path}: {e}")

# Factory function
def create_document_loader(file_path: str) -> DocumentLoaderInterface:
    if file_path.lower().endswith('.pdf'):
        return PDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")