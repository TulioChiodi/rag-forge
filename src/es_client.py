import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch
from zoneinfo import ZoneInfo

from src.config import settings
from src.logging_conf import logger


class ESClient:
    _instance: Optional[AsyncElasticsearch] = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        """Get or create Elasticsearch client instance."""
        if not cls._instance:
            async with cls._lock:
                if not cls._instance:
                    cls._instance = AsyncElasticsearch(
                        hosts=[settings.ES_URL], request_timeout=30, max_retries=3
                    )
                    # Create index if it doesn't exist
                    await cls.create_index()
        return cls._instance

    @classmethod
    async def create_index(cls) -> None:
        """Create the index with proper mappings if it doesn't exist."""
        try:
            if not await cls._instance.indices.exists(index=settings.ES_INDEX):
                await cls._instance.indices.create(
                    index=settings.ES_INDEX,
                    body={
                        "mappings": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "content": {"type": "text"},
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": settings.EMBEDDING_DIMENSIONS,
                                    "index": True,
                                    "similarity": "cosine",
                                },
                                "source": {"type": "keyword"},
                            }
                        }
                    },
                )
                logger.info(
                    f"Created index {settings.ES_INDEX} with vector search mappings"
                )
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    @classmethod
    async def close(cls):
        """Close Elasticsearch client connection."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

    @classmethod
    async def is_index_empty(cls) -> bool:
        """Check if the index is empty or doesn't exist.

        Returns:
            bool: True if index doesn't exist or is empty, False otherwise
        """
        try:
            client = await cls.get_client()
            # First check if index exists
            if not await client.indices.exists(index=settings.ES_INDEX):
                return True

            # Then check document count
            result = await client.count(index=settings.ES_INDEX)
            return result["count"] == 0
        except Exception as e:
            logger.error(f"Error checking if index is empty: {str(e)}")
            return True  # Assume empty on error for safety

    @classmethod
    async def check_health(cls) -> Dict[str, Any]:
        """Check Elasticsearch and index health."""
        try:
            client = await cls.get_client()

            # Basic cluster health check
            cluster_health = await client.cluster.health()
            index_exists = await client.indices.exists(index=settings.ES_INDEX)

            if not index_exists:
                return {
                    "status": "red",
                    "message": f"Index {settings.ES_INDEX} does not exist",
                    "timestamp": datetime.now(
                        ZoneInfo("America/Sao_Paulo")
                    ).isoformat(),
                }

            return {
                "status": cluster_health["status"],
                "message": "Service is healthy"
                if cluster_health["status"] in ["green", "yellow"]
                else "Service is degraded",
                "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "red",
                "message": "Service is unavailable",
                "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            }


@asynccontextmanager
async def get_es_client():
    """Context manager for getting ES client."""
    try:
        client = await ESClient.get_client()
        yield client
    except Exception as e:
        logger.error(f"Error with ES client: {str(e)}")
        raise
