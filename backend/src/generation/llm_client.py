from abc import ABC, abstractmethod
from typing import List, Optional
import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

class LLMClientInterface(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate response from simple text prompt"""
        pass

class DeepSeekLLMClient(LLMClientInterface):
    """DeepSeek LLM client implementation"""
    
    def __init__(self, 
                 model: str = "deepseek-chat",
                 temperature: float = 0.1,
                 api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        try:
            self._llm = init_chat_model(
                self.model,
                model_provider="deepseek",
                api_key=self.api_key,
               # model_kwargs={"temperature": self.temperature}
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize DeepSeek LLM: {e}")
    
    def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response from messages"""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def generate_text(self, prompt: str) -> str:
        """Generate response from simple text prompt"""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        messages = [HumanMessage(content=prompt)]
        return self.generate_response(messages)

class OpenAILLMClient(LLMClientInterface):
    """OpenAI LLM client implementation"""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI LLM: {e}")
    
    def generate_response(self, messages: List[BaseMessage]) -> str:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def generate_text(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        messages = [HumanMessage(content=prompt)]
        return self.generate_response(messages)

# Factory function
def create_llm_client(provider: str = "deepseek", **kwargs) -> LLMClientInterface:
    """Factory function to create LLM clients"""
    if provider.lower() == "deepseek":
        return DeepSeekLLMClient(**kwargs)
    elif provider.lower() == "openai":
        return OpenAILLMClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")