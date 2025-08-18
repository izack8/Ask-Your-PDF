class RAGPromptTemplate:
    """Prompt template for RAG queries"""

    def create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM"""
        return f"Question: {question}\nContext: {context}\nAnswer:"
    
    

