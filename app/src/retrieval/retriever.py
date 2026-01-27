from src.retrieval.vector_store import VectorStoreInterface
from src.utils.utils import extract_links

class Retriever:
    def __init__(self, vector_store: VectorStoreInterface, reranker=None):
        self.vector_store = vector_store
        self.reranker = reranker
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieve documents relevant to the query"""
        
        docs = self.vector_store.similarity_search(query, k=k*2)
        
        if self.reranker:
            docs = self.reranker.rerank(query, docs)
    
        return docs[:k]
    
    def retrieve_api(self, query: str, k: int = 5):
        """Retrieve API links instead of documents"""
        try:
            docs = self.retrieve(query, k)[:k]
            links = []
            for doc in docs:
                if 'content' in doc and doc['content']:
                    links.extend(extract_links(doc['content']))
            return list(set(links)) 
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {e}")

