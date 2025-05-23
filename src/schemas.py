from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the documents")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question based on the documents")
    chunks: List[str] = Field(..., description="The chunks used to generate the answer")

class DocumentResponse(BaseModel):
    message: str = Field(..., description="Status message")
    documents_indexed: int = Field(..., description="Number of documents successfully indexed")
    total_chunks: int = Field(..., description="Total number of chunks created from the documents")
    failed_files: List[str] = Field(default=[], description="List of files that failed to process")

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status (green, yellow, red)")
    message: str = Field(..., description="Health status message")
    timestamp: str = Field(..., description="Timestamp of the health check") 