from typing import List, Tuple
import asyncio
from elasticsearch import NotFoundError
from langchain_openai import OpenAIEmbeddings
from src.config import settings
from src.logging_conf import logger
from src.llm_providers.factory import create_llm_provider
from src.es_client import get_es_client
from src.es_client import ESClient

async def retrieve_context(question: str, top_k: int = 5) -> List[str]:
    """Retrieve relevant context from Elasticsearch based on the question."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
        
    logger.info(f"Retrieving context for question - Length: {len(question)}, top_k: {top_k}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    
    try:
        embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
        question_vector = await asyncio.wait_for(
            embeddings.aembed_query(question),
            timeout=10.0  # 10 second timeout for embeddings
        )
        
        async with get_es_client() as client:
            try:
                response = await asyncio.wait_for(
                    client.search(
                        index=settings.ES_INDEX,
                        knn={
                            "field": "vector",
                            "query_vector": question_vector,
                            "k": top_k,
                            "num_candidates": top_k * 2
                        },
                        size=top_k,
                        request_timeout=30
                    ),
                    timeout=35.0  # 35 second timeout (slightly longer than ES timeout)
                )
            except NotFoundError:
                logger.warning(f"Index {settings.ES_INDEX} not found")
                return []
                
            # Extract the content from the search results
            contexts = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
        
        if not contexts:
            logger.warning("No relevant contexts found")
            return []
            
        avg_length = sum(len(c) for c in contexts) / len(contexts)
        logger.info(f"Context retrieved successfully - {len(contexts)} contexts, avg length: {avg_length:.0f}")
        
        return contexts
        
    except asyncio.TimeoutError:
        logger.error("Timeout while retrieving context")
        raise
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise

async def generate_answer(question: str, contexts: List[str]) -> str:
    """Generate an answer based on the question and retrieved contexts."""
    if not contexts:
        return "I apologize, but I don't have enough context to answer your question accurately."
        
    logger.info("Generating answer.")
    llm_provider = create_llm_provider()
    
    prompt = f"""Answer the question based only on the following context. If the context doesn't contain enough information to answer accurately, say so.

Context:
{' '.join(contexts)}

Question: {question}

Answer:"""
    
    system_message = "You are a helpful assistant that provides accurate answers based only on the provided context. If you cannot answer based on the context, say so clearly."
    
    try:
        answer = await asyncio.wait_for(
            llm_provider.generate_completion(prompt, system_message),
            timeout=30.0  # 30 second timeout for LLM
        )
        return answer
    except asyncio.TimeoutError:
        logger.error("Timeout while generating answer")
        return "I apologize, but I was unable to generate an answer in time. Please try again."
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

async def rag_answer(question: str) -> Tuple[str, List[str]]:
    """Retrieve context and generate an answer for the given question."""
    logger.info("Starting RAG process.")
    
    try:
        # Check if index is empty
        if await ESClient.is_index_empty():
            return "I apologize, but there are no documents in the knowledge base yet. Please upload some PDF documents first so I can help answer your questions.", []
            
        contexts = await retrieve_context(question)
        answer = await generate_answer(question, contexts)
        logger.info(f"RAG process completed successfully - Answer length: {len(answer)}, Chunks used: {len(contexts)}")
        return answer, contexts
    except Exception as e:
        logger.error(f"Error in RAG process: {str(e)}", exc_info=True)
        raise