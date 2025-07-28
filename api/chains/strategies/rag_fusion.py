"""RAG Fusion retrieval strategy implementation using LCEL pattern."""

from typing import List, Dict, Any, Optional
import logging
import asyncio
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)

# Template for generating multiple queries
MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate 4 different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines. Only provide the questions, no numbering or additional text.

Original question: {question}"""

# Template for final answer generation
RAG_TEMPLATE = """Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""


class RAGFusionRetrieval(BaseRetrievalStrategy):
    """RAG Fusion retrieval strategy using multi-query generation and Reciprocal Rank Fusion."""
    
    name = "rag_fusion_retrieval"
    
    def __init__(self, k: int = 5, k_per_query: int = 10, rrf_k: int = 60):
        """Initialize RAG Fusion retrieval.
        
        Args:
            k: Number of final documents to return
            k_per_query: Number of documents to retrieve per generated query
            rrf_k: RRF parameter (higher = more conservative fusion)
        """
        super().__init__(k=k)
        self.k_per_query = k_per_query
        self.rrf_k = rrf_k
        logger.info(f"🏗️ Initializing {self.name} with k={k}, k_per_query={k_per_query}, rrf_k={rrf_k}")
    
    async def setup(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        document: Optional[Document] = None,
        **kwargs
    ):
        """Setup RAG Fusion chain components."""
        logger.info(f"🔧 {self.name} setup: Initializing multi-query generation and RRF components")
        
        if not isinstance(vector_store, VectorStore):
            raise ValueError("vector_store must be an instance of VectorStore")
        
        # Store vector store for retrieval
        self._vector_store = vector_store
        self._base_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.k_per_query}
        )
        
        # Create LLM for multi-query generation
        self._llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            **kwargs
        )
        
        # Create prompts
        self._multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
        self._rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        
        # Create output parser
        self._parser = StrOutputParser()
        
        logger.info(f"📝 {self.name} setup: Multi-query generation with {model}")
        logger.info(f"🔍 {self.name} setup: Base retriever configured for {self.k_per_query} docs per query")
        
        logger.info(f"🎯 {self.name} setup completed successfully! Ready to process queries with RRF scoring.")
    
    async def _generate_queries(self, query: str) -> List[str]:
        """Generate multiple queries from the original question."""
        logger.info(f"🔄 Step 1: Generating alternative queries for: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Create query generation chain
            generation_chain = self._multi_query_prompt | self._llm | self._parser
            
            # Generate queries
            result = await generation_chain.ainvoke({"question": query})
            
            # Parse queries
            queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
            
            # Always include the original query
            if query not in queries:
                queries.insert(0, query)
            
            logger.info(f"✅ Generated {len(queries)} total queries (including original)")
            logger.info(f"🔍 Queries: {queries}")
            return queries
            
        except Exception as e:
            logger.warning(f"⚠️ Query generation failed: {e}, using original query only")
            return [query]
    
    async def _retrieve_and_fuse_step(self, queries: List[str]) -> List[Document]:
        """Step 2: Retrieve documents for all queries and apply RRF fusion."""
        logger.info(f"🔍 Step 2: Running parallel retrieval for {len(queries)} queries")
        
        # Create retrieval tasks for all queries
        retrieval_tasks = [
            self._retrieve_for_query(query) for query in queries
        ]
        
        # Execute retrievals in parallel
        all_results = await asyncio.gather(*retrieval_tasks)
        
        # Log retrieval results
        total_retrieved = sum(len(docs) for docs in all_results)
        logger.info(f"📄 Retrieved {total_retrieved} total documents across all queries")
        
        # Apply RRF fusion
        fused_docs = self._apply_rrf_fusion(all_results)
        logger.info(f"🎯 RRF fusion produced {len(fused_docs)} final documents")
        
        return fused_docs
    
    async def _retrieve_for_query(self, query: str) -> List[Document]:
        """Retrieve documents for a single query."""
        try:
            docs = await self._base_retriever.aget_relevant_documents(query)
            logger.debug(f"📋 Query '{query[:50]}...' retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.warning(f"⚠️ Query '{query[:50]}...' failed: {e}")
            return []
    
    def _apply_rrf_fusion(self, all_results: List[List[Document]]) -> List[Document]:
        """Apply Reciprocal Rank Fusion to combine results from multiple queries."""
        # Track document scores using content + metadata as key
        doc_scores = defaultdict(float)
        doc_registry = {}  # Store actual document objects
        
        for query_idx, documents in enumerate(all_results):
            for rank, doc in enumerate(documents):
                # Create unique key for document
                doc_key = f"{doc.page_content}:{doc.metadata.get('document_id', '')}"
                
                # Calculate RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                doc_scores[doc_key] += rrf_score
                
                # Store document object (if not already stored)
                if doc_key not in doc_registry:
                    doc_registry[doc_key] = doc
        
        # Sort documents by RRF score (highest first)
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k documents
        final_docs = [doc_registry[doc_key] for doc_key, score in sorted_docs[:self.k]]
        
        logger.info(f"⚖️ RRF scoring: Combined {len(doc_registry)} unique documents, returning top {len(final_docs)}")
        
        return final_docs
    
    async def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """Step 3: Generate final answer using retrieved documents."""
        logger.info(f"🎯 Step 3: Generating answer using {len(documents)} context documents")
        
        try:
            # Format documents into context
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Create answer generation chain
            answer_chain = self._rag_prompt | self._llm | self._parser
            
            # Generate answer
            answer = await answer_chain.ainvoke({
                "context": context,
                "question": query
            })
            
            logger.info(f"✅ Generated answer ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents using RAG Fusion approach."""
        logger.info(f"🔍 Running {self.name} retriever for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Step 1: Generate multiple queries
        queries = await self._generate_queries(query)
        
        # Step 2: Retrieve and fuse documents
        documents = await self._retrieve_and_fuse_step(queries)
        
        logger.info(f"📄 {self.name} returned {len(documents)} fused documents")
        return documents
    
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run RAG Fusion retrieval and answer generation."""
        logger.info(f"🚀 Starting {self.name} manual chain execution for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Step 1: Generate multiple queries
            queries = await self._generate_queries(query)
            
            # Step 2: Retrieve and fuse documents
            documents = await self._retrieve_and_fuse_step(queries)
            
            # Step 3: Generate final answer
            answer = await self._generate_answer(query, documents)
            
            logger.info(f"✅ {self.name} completed successfully with multi-query RRF fusion")
            
            return {
                "answer": answer,
                "strategy": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name} execution failed: {e}")
            raise