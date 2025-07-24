from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def format_docs(docs):
    """Format documents into a single string."""
    print("docs", docs)
    return "\n\n".join(doc.get("page_content", "") for doc in docs)

def create_chain(retriever, rag_prompt, chat_model):
    """Create a RAG chain that retrieves documents and generates a response."""
    return (
        # First get the retriever chain and format the documents
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question")
        }
        # Then send to LLM
        | rag_prompt 
        | chat_model
    )

def format_docs(docs):
    """Format documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

class NaiveRetrievalChain:
    """Chain for naive retrieval using LangChain's ConversationalRetrievalChain"""
    
    def __init__(self, openai_api_key: str, vector_store: Any):
        self.qa_template = PromptTemplate(
            template=(
                "You are a helpful AI assistant that answers questions based on the "
                "provided PDF document. Always be truthful and base your answers on "
                "the context provided. If you don't know something or if it's not "
                "in the context, say so. When you use information from the context, "
                "try to cite the specific parts you're referring to.\n\n"
                "Context: {context}\n\n"
                "Question: {question}"
            ),
            input_variables=["context", "question"]
        )

        self.rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        self.chat_model = ChatOpenAI(model="gpt-4.1-nano", openai_api_key=openai_api_key)
        
        self.chain = create_chain(vector_store.as_retriever(search_kwargs={"k": 5}), self.rag_prompt, self.chat_model)
    
    async def run(self, question: str) -> Dict[str, Any]:
        """Run the chain on a question"""
        print("Running chain with question:", question)
        result = await self.chain.ainvoke({"question": question})
        print("Chain execution completed")

        
        return {
            "answer": result.content,  # ChatMessage object has content attribute
            "strategy": "naive_retrieval"
        } 