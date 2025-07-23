from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_template}
        )
    
    async def run(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run the chain on a question"""
        # Convert chat history to the format expected by the chain
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    self.memory.chat_memory.add_user_message(message["content"])
                else:
                    self.memory.chat_memory.add_ai_message(message["content"])
        
        # Run the chain
        result = await self.chain.acall({"question": question})
        
        # Extract source documents
        source_docs = [
            f"Page {doc.metadata['page']}: {doc.page_content[:200]}..."
            for doc in result.get("source_documents", [])
        ]
        
        return {
            "answer": result["answer"],
            "sources": source_docs,
            "strategy": "naive_retrieval"
        } 