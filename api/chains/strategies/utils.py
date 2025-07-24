"""Utility functions for retrieval strategies."""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Template for RAG responses
RAG_TEMPLATE = """You are a helpful AI assistant answering questions based on the provided context.

Context:
{context}

Question:
{question}

Please provide a clear, accurate, and helpful response based on the context above.
If the context doesn't contain enough information to answer the question fully,
acknowledge this and provide the best possible answer with the available information.

Response:"""

def create_rag_chain(
    retriever: Any,
    openai_api_key: str,
    model: str = "gpt-4-1106-preview",
    temperature: float = 0,
    **kwargs
) -> Any:
    """Create a RAG chain with the given retriever.
    
    Args:
        retriever: Document retriever component
        openai_api_key: OpenAI API key
        model: Model to use for response generation
        temperature: Temperature for response generation
        **kwargs: Additional parameters for chain creation
        
    Returns:
        Configured RAG chain
    """
    # Create prompt and model
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    llm = ChatOpenAI(
        temperature=temperature,
        model=model,
        openai_api_key=openai_api_key,
        **kwargs
    )
    
    # Create and return chain
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
    
def _format_docs(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a string.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted string of document contents
    """
    return "\n\n".join(doc.page_content for doc in docs) 