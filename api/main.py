"""
FastAPI Application for CVS HealthHub AI
REST API with endpoints for chat, search, health checks, and metrics
"""
import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    ChatRequest, ChatResponse, SearchRequest, SearchResponse,
    HealthCheckResponse, MetricsResponse, ErrorResponse, SearchResult,
    ImageAnalysisRequest, ImageAnalysisResponse
)
from agent.healthhub_agent import chat as agent_chat, vector_store, analyze_prescription_image
from dotenv import load_dotenv

load_dotenv()

# Global metrics
metrics = {
    "total_requests": 0,
    "total_tool_calls": 0,
    "start_time": time.time()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    # Startup
    print(" CVS HealthHub AI API starting...")
    print(f"📊 Vector Store: {vector_store.get_collection_stats()}")
    yield
    # Shutdown
    print("👋 CVS HealthHub AI API shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="CVS HealthHub AI API",
    description="Enterprise RAG system for healthcare information, prescriptions, and appointments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    metrics["total_requests"] += 1
    return response


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CVS HealthHub AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns service status and vector store statistics
    """
    try:
        vs_stats = vector_store.get_collection_stats()
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            vector_store_status=vs_stats
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Get API metrics
    Returns request counts, tool calls, and vector store statistics
    """
    try:
        vs_stats = vector_store.get_collection_stats()
        return MetricsResponse(
            total_requests=metrics["total_requests"],
            total_tool_calls=metrics["total_tool_calls"],
            vector_store_stats=vs_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint
    Process user messages through the healthcare agent

    Example:
    ```json
    {
        "message": "What vaccines are available?",
        "chat_history": []
    }
    ```
    """
    try:
        # Process through agent
        result = agent_chat(
            user_input=request.message,
            chat_history=request.chat_history or []
        )

        # Update metrics
        metrics["total_tool_calls"] += result.get("tool_calls", 0)

        return ChatResponse(
            response=result["response"],
            tool_calls=result.get("tool_calls", 0),
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_endpoint(request: SearchRequest):
    """
    Semantic search endpoint
    Search the vector database for relevant information

    Example:
    ```json
    {
        "query": "blood pressure medications",
        "k": 5,
        "category": "medication"
    }
    ```
    """
    try:
        # Build filter
        filter_dict = None
        if request.category:
            filter_dict = {"category": request.category}

        # Perform search
        results = vector_store.similarity_search(
            query=request.query,
            k=request.k,
            filter_dict=filter_dict
        )

        # Format results
        formatted_results = [
            SearchResult(
                content=r["content"],
                metadata=r.get("metadata", {}),
                distance=r.get("distance"),
                id=r["id"]
            )
            for r in results
        ]

        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )


@app.post("/analyze-image", response_model=ImageAnalysisResponse, tags=["Vision"])
async def analyze_image_endpoint(request: ImageAnalysisRequest):
    """
    Analyze prescription or medication images using NVIDIA's multimodal model
    Extracts medication info, dosage, and provides safety recommendations

    Example:
    ```json
    {
        "image_base64": "base64_encoded_image_string"
    }
    ```
    """
    try:
        # Analyze image
        result = analyze_prescription_image(
            image_data=request.image_base64,
            image_type="base64"
        )

        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Image analysis failed")
            )

        # Update metrics
        metrics["total_requests"] += 1

        return ImageAnalysisResponse(
            analysis=result["analysis"],
            medications_detected=result.get("medications_detected", []),
            has_additional_info=result.get("has_additional_info", False),
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            success=False
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            success=False
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🏥 Starting CVS HealthHub AI API on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )