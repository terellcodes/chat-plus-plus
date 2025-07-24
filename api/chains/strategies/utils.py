"""Utility functions for retrieval strategies."""

from typing import List, Dict, Any, Union
import logging
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

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
    logger.info(f"Creating RAG chain with model {model}")
    
    # Create prompt and model
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    llm = ChatOpenAI(
        temperature=temperature,
        model=model,
        openai_api_key=openai_api_key,
        **kwargs
    )
    
    # Create chain using LCEL
    chain = (
        {
            "context": itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain created successfully")
    return chain
    
def _format_docs(docs: Union[List[Document], List[Dict[str, Any]]]) -> str:
    """Format retrieved documents into a string.
    
    Args:
        docs: List of retrieved documents or dictionaries
        
    Returns:
        Formatted string of document contents
    """
    if not docs:
        logger.warning("No documents provided to format")
        return ""
        
    logger.debug(f"Formatting {len(docs)} documents")
    logger.debug(f"First document type: {type(docs[0]).__name__}")
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        try:
            if isinstance(doc, Document):
                text = doc.page_content
            elif isinstance(doc, dict):
                text = doc.get("page_content", "")
            else:
                logger.warning(f"Unexpected document type at index {i}: {type(doc).__name__}")
                continue
                
            if text and isinstance(text, str):
                formatted_docs.append(text)
            else:
                logger.warning(f"Invalid or empty text at index {i}: {type(text).__name__}")
                
        except Exception as e:
            logger.error(f"Error formatting document at index {i}: {str(e)}")
            continue
            
    if not formatted_docs:
        logger.warning("No valid documents found after formatting")
        return ""
        
    logger.debug(f"Successfully formatted {len(formatted_docs)} documents")
    return "\n\n".join(formatted_docs) 