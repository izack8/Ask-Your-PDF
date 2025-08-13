from .vector_store import VectorStoreInterface

class Retriever:
    def __init__(self, vector_store: VectorStoreInterface, reranker=None):
        self.vector_store = vector_store
        self.reranker = reranker
    
    def retrieve(self, query: str, k: int = 5):
        # 1. Semantic search
        docs = self.vector_store.similarity_search(query, k=k*2)
        
        # 2. Optional re-ranking
        if self.reranker:
            docs = self.reranker.rerank(query, docs)
        
        # 3. Filter and return top k
        return docs[:k]