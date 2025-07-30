import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Tuple

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.es_client import ESClient
from src.ingest import process_document
from src.logging_conf import logger
from src.rag import rag_answer
from src.schemas import (
    DocumentResponse,
    HealthCheckResponse,
    QuestionRequest,
    QuestionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await ESClient.get_client()
        logger.info("Successfully initialized Elasticsearch client")
        yield
    finally:
        await ESClient.close()
        logger.info("Closed Elasticsearch client connection")


app = FastAPI(
    title="RAG-Forge API",
    description="""
PDF RAG (Retrieval Augmented Generation) API

This API allows you to:
• Upload PDF documents for processing and indexing
• Ask questions about the uploaded documents using natural language
• Check system health and status

Key Features:
• Document Processing: Supports multiple PDF uploads with concurrent processing
• Question Answering: Uses RAG to provide accurate answers based on uploaded documents
• Health Monitoring: Real-time system health checks
    """,
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "System health monitoring endpoints"},
        {
            "name": "Documents",
            "description": "Document upload and processing endpoints",
        },
        {"name": "Questions", "description": "Question answering endpoints using RAG"},
    ],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "examples": {
                        "healthy": {
                            "summary": "Healthy System",
                            "value": {
                                "status": "green",
                                "message": "Service is healthy",
                                "timestamp": "2024-03-20T10:00:00-03:00",
                            },
                        },
                        "degraded": {
                            "summary": "Degraded System",
                            "value": {
                                "status": "yellow",
                                "message": "Service is degraded",
                                "timestamp": "2024-03-20T10:00:00-03:00",
                            },
                        },
                    }
                }
            },
        },
        503: {
            "description": "System is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "status": "red",
                            "message": "Service is unavailable",
                            "timestamp": "2024-03-20T10:00:00-03:00",
                        }
                    }
                }
            },
        },
    },
)
async def health_check():
    """Check system health status.

    Checks the health of the system components including:
    - Elasticsearch connection and cluster status
    - Index existence and status

    Returns:
        HealthCheckResponse: Health status information

    Raises:
        HTTPException: 503 error if the system is in an unhealthy state
    """
    health_info = await ESClient.check_health()

    if health_info["status"] == "red":
        raise HTTPException(status_code=503, detail=health_info)

    return HealthCheckResponse(**health_info)


@app.post(
    "/documents",
    response_model=DocumentResponse,
    tags=["Documents"],
    responses={
        200: {
            "description": "Documents processed successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "All documents processed successfully",
                            "value": {
                                "message": "Documents processed successfully",
                                "documents_indexed": 2,
                                "total_chunks": 45,
                                "failed_files": [],
                            },
                        },
                        "partial_success": {
                            "summary": "Some documents failed",
                            "value": {
                                "message": "Some documents failed to process",
                                "documents_indexed": 1,
                                "total_chunks": 20,
                                "failed_files": ["invalid.txt (not a PDF)"],
                            },
                        },
                    }
                }
            },
        },
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No documents were processed successfully. Failures: manual1.pdf (processing error)"
                    }
                }
            },
        },
    },
)
async def upload_documents(
    files: List[UploadFile] = File(  # noqa: B008
        ...,
        description="One or more PDF files to upload and process. Only PDF files are accepted.",
    ),
):
    """Upload and process PDF documents.

    The documents will be:
    1. Validated for PDF format
    2. Split into chunks for better processing
    3. Embedded using OpenAI's text-embedding-3-large model
    4. Indexed in Elasticsearch for retrieval

    Example Usage with curl:
    ```bash
    curl -X POST "http://localhost:8000/documents" \\
         -H "accept: application/json" \\
         -H "Content-Type: multipart/form-data" \\
         -F "files=@manual1.pdf" \\
         -F "files=@manual2.pdf"
    ```

    Returns:
        DocumentResponse: Processing results including success/failure information

    Raises:
        HTTPException: 400 error if no documents were processed successfully
    """
    logger.info(
        f"Document upload request received - {len(files)} files: {[f.filename for f in files]}"
    )

    if not files:
        return DocumentResponse(
            message="No documents were provided. Please upload at least one PDF file.",
            documents_indexed=0,
            total_chunks=0,
            failed_files=[],
        )

    failed_files = []
    valid_files = []

    # First validate all files
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            failed_files.append(f"{file.filename} (not a PDF)")
            continue
        valid_files.append(file)

    if not valid_files:
        return DocumentResponse(
            message="No valid PDF files were provided. Please upload PDF files only.",
            documents_indexed=0,
            total_chunks=0,
            failed_files=failed_files,
        )

    async def process_single_file(file: UploadFile) -> Tuple[int, int]:
        try:
            async with aiofiles.tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                content = await file.read()
                await tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                docs, chunks = await process_document(tmp_path)
                logger.info(f"Successfully processed {file.filename}: {chunks} chunks")
                return docs, chunks
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                failed_files.append(f"{file.filename} (processing error)")
                return 0, 0
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error handling {file.filename}: {str(e)}")
            failed_files.append(f"{file.filename} (upload error)")
            return 0, 0

    # Process all valid files concurrently
    results = await asyncio.gather(*[process_single_file(file) for file in valid_files])

    # Aggregate results
    total_docs = sum(docs for docs, _ in results)
    total_chunks = sum(chunks for _, chunks in results)

    logger.info(
        f"Document upload completed - {total_docs} docs, {total_chunks} chunks. "
        f"Failed files: {len(failed_files)}"
    )

    if total_docs == 0 and failed_files:
        raise HTTPException(
            status_code=400,
            detail=f"No documents were processed successfully. Failures: {', '.join(failed_files)}",
        )

    return DocumentResponse(
        message="Documents processed successfully"
        if not failed_files
        else "Some documents failed to process",
        documents_indexed=total_docs,
        total_chunks=total_chunks,
        failed_files=failed_files,
    )


@app.post(
    "/question",
    response_model=QuestionResponse,
    tags=["Questions"],
    responses={
        200: {
            "description": "Question answered successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "detailed_answer": {
                            "summary": "Detailed answer with context",
                            "value": {
                                "answer": "According to the maintenance manual, the equipment requires monthly inspections of all moving parts, quarterly calibration of sensors, and annual replacement of filters. The maintenance procedures should be performed by certified technicians.",
                                "chunks": [
                                    "Monthly maintenance checklist: 1. Inspect all moving parts for wear and tear 2. Lubricate bearings and joints 3. Check and record sensor readings",
                                    "Quarterly maintenance requirements include full calibration of all sensors and verification of safety systems.",
                                    "Annual maintenance must be performed by certified technicians and includes complete filter replacement and system overhaul.",
                                ],
                            },
                        },
                        "no_context": {
                            "summary": "Answer when no relevant context found",
                            "value": {
                                "answer": "I could not find specific information about maintenance procedures in the uploaded documents. Please ensure relevant maintenance manuals have been uploaded.",
                                "chunks": [],
                            },
                        },
                    }
                }
            },
        },
    },
)
async def ask_question(request: QuestionRequest):
    """Ask a natural language question about the uploaded documents.

    The system will:
    1. Convert your question into an embedding
    2. Find the most relevant document chunks
    3. Use RAG with a language model to generate an accurate answer

    Example Usage with curl:
    ```bash
    curl -X POST "http://localhost:8000/question" \\
         -H "accept: application/json" \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What are the maintenance procedures for the equipment?"}'
    ```

    Example Usage with Swagger UI:
    ```json
    {
      "question": "What are the maintenance procedures for the equipment?"
    }
    ```

    Note:
        Documents must be uploaded first using the /documents endpoint before asking questions.

    Args:
        request (QuestionRequest): The question to ask about the documents

    Returns:
        QuestionResponse: The answer and relevant document chunks used

    Raises:
        HTTPException: 500 error if there's an internal error processing the question
    """
    logger.info(f"Question received - Length: {len(request.question)}")

    try:
        answer, chunks = await rag_answer(request.question)
        logger.info(
            f"Question answered successfully - Answer length: {len(answer)}, Chunks used: {len(chunks)}"
        )
        return QuestionResponse(answer=answer, chunks=chunks)
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
