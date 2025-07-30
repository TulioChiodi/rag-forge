import asyncio
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from elasticsearch import NotFoundError
from elasticsearch.helpers import async_bulk
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.es_client import get_es_client
from src.logging_conf import logger
from src.utils import extract_text_from_pdf, normalize_text, run_ocrmypdf


async def process_document(file_path: Path) -> Tuple[int, int]:
    """Process a document from file to chunks and index them.

    Args:
        file_path: Path to the PDF file to process

    Returns:
        Tuple[int, int]: (number of documents, number of chunks)
        Returns (0, 0) if processing fails
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 0, 0

    logger.info(f"Starting document processing for {file_path.name}")

    try:
        # Extract text with timeout
        text = await asyncio.wait_for(
            extract_text_from_pdf(file_path),
            timeout=30.0,  # 30 seconds for PDF extraction
        )

        if text is None:
            logger.info("No text found in PDF, falling back to OCR.")
            out_txt_path = file_path.with_suffix(".txt")
            # OCR with timeout
            text = await asyncio.wait_for(
                run_ocrmypdf(file_path, out_txt_path),
                timeout=120.0,  # 2 minutes for OCR
            )

        if not text:
            logger.error("Failed to extract text from document")
            return 0, 0

        normalized_text = normalize_text(text)

        doc = Document(
            page_content=normalized_text, metadata={"source": file_path.name}
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        chunks = splitter.split_documents([doc])

        if not chunks:
            logger.warning("Document was split into 0 chunks")
            return 0, 0

        logger.info(f"Document split into chunks. # of chunks: {len(chunks)}")

        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY
        )
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings with timeout
        vectors = await asyncio.wait_for(
            embeddings.aembed_documents(texts),
            timeout=60.0,  # 1 minute for embeddings
        )

        # Insert chunks with timeout
        await asyncio.wait_for(
            insert_chunks(chunks, vectors),
            timeout=60.0,  # 1 minute for insertion
        )

        return 1, len(chunks)

    except asyncio.TimeoutError:
        logger.error(f"Timeout processing document {file_path.name}")
        return 0, 0
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return 0, 0


async def insert_chunks(docs: List[Document], vectors: List[List[float]]) -> None:
    """Index document chunks with their vectors in Elasticsearch.

    Args:
        docs: List of document chunks to index
        vectors: List of vector embeddings corresponding to the chunks

    Raises:
        ValueError: If inputs are invalid
        NotFoundError: If ES index doesn't exist
        Exception: If bulk insertion fails
    """
    if not docs or not vectors or len(docs) != len(vectors):
        raise ValueError(
            "Invalid input: docs and vectors must be non-empty and same length"
        )

    async with get_es_client() as client:
        actions = [
            {
                "_index": settings.ES_INDEX,
                "_source": {
                    "id": str(uuid4()),
                    "content": doc.page_content,
                    "vector": vec,
                    **doc.metadata,
                },
            }
            for doc, vec in zip(docs, vectors)
        ]

        try:
            result = await async_bulk(
                client,
                actions,
                chunk_size=100,
                max_retries=3,
                request_timeout=30,
                raise_on_error=False,  # Get error details instead of failing fast
            )

            success, failed = result

            if isinstance(failed, list):
                n_failed = len(failed)
                if failed:
                    logger.error(f"Failed documents: {failed}")
            else:
                n_failed = failed

            logger.info(
                f"Bulk insert completed: {success} succeeded, {n_failed} failed"
            )

            if n_failed > 0:
                raise Exception(
                    f"Bulk insert failed: {n_failed} documents failed to index"
                )

        except NotFoundError:
            logger.error(f"Index {settings.ES_INDEX} not found")
            raise
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise
