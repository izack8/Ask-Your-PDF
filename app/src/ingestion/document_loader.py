from abc import ABC, abstractmethod
from typing import List
import fitz
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
    

class ExcelLoader(DocumentLoaderInterface):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        try:
            import pandas as pd
            df = pd.read_excel(self.file_path)
            documents = []
            
            for index, row in df.iterrows():
                content = row.to_string()
                metadata = {
                    "source": self.file_path,
                    "row": index + 1
                }
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return documents
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Excel {self.file_path}: {e}")
        
class DocumentLoader:
    def __init__(self, loader: DocumentLoaderInterface):
        self.loader = loader
    
    def load_documents(self) -> List[Document]:
        """Load documents using the specified loader"""
        try:
            return self.loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load documents: {e}")
    
    def set_loader(self, loader: DocumentLoaderInterface):
        """Set a new document loader"""
        self.loader = loader

# Factory function
def create_document_loader(file_path: str) -> DocumentLoaderInterface:
    if file_path.lower().endswith('.pdf'):
        return DocumentLoader(PDFLoader(file_path))
    elif file_path.lower().endswith(('.xls', '.xlsx')):
        return DocumentLoader(ExcelLoader(file_path))