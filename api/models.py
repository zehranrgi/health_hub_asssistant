"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question", min_length=1)
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation history"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What vaccines are available?",
                "chat_history": []
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Agent's response")
    tool_calls: int = Field(default=0, description="Number of tools called")
    success: bool = Field(default=True, description="Request success status")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "We have the following vaccines available: Flu vaccine, COVID-19 vaccine...",
                "tool_calls": 1,
                "success": True
            }
        }


class SearchRequest(BaseModel):
    """Request model for semantic search endpoint"""
    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    category: Optional[str] = Field(
        default=None,
        description="Filter by category (medication, vaccines, services, insurance, interactions)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "blood pressure medications",
                "k": 5,
                "category": "medication"
            }
        }


class SearchResult(BaseModel):
    """Individual search result"""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default={}, description="Document metadata")
    distance: Optional[float] = Field(default=None, description="Similarity distance")
    id: str = Field(..., description="Document ID")


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    success: bool = Field(default=True, description="Request success status")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "content": "Lisinopril is used to treat high blood pressure...",
                        "metadata": {"category": "medication", "source": "FDA"},
                        "distance": 0.23,
                        "id": "doc_123_chunk_0"
                    }
                ],
                "total": 1,
                "success": True
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    vector_store_status: Dict[str, Any] = Field(..., description="Vector store statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_status": {
                    "collection_name": "healthhub_knowledge",
                    "total_chunks": 150
                }
            }
        }


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    total_requests: int = Field(..., description="Total API requests")
    total_tool_calls: int = Field(..., description="Total agent tool calls")
    vector_store_stats: Dict[str, Any] = Field(..., description="Vector store statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 42,
                "total_tool_calls": 28,
                "vector_store_stats": {
                    "total_chunks": 150,
                    "collection_name": "healthhub_knowledge"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    success: bool = Field(default=False, description="Request success status")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Message field is required",
                "success": False
            }
        }
