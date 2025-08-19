class RAGPromptTemplate:
    """Prompt template for RAG queries"""

    def create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM"""
        return f"""
        You are a helpful assistant, that will talk about Isaac's work experiences in Software Engineering / Data Science. You will answer the user's question based on the provided context. If the question does not have enough information, you can say "I'm sorry, I'm not sure! You can reach out to isaactaypb@gmail.com for more details, and he would be happy to have a conversation with you!".
        Question: {question}\nContext: {context}\nAnswer:"""

    def test_prompt(self, question: str, context: str) -> None:
        """Test the prompt generation"""
        prompt = f"""
                repeat the below
                {question}{context}
                    """
        return prompt