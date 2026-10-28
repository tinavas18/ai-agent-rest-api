
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import run_research

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AI Research Agent API",
    description="A REST API for a Gemini-powered AI research assistant.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# Validate incoming requests
class ResearchRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=1000,
        description="The topic or question to research",
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AI Research Agent API is running",
        "docs": "/docs",
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Research endpoint with rate limiting and error handling
@app.post("/research")
@limiter.limit("5/minute")
def research(request: Request, payload: ResearchRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured.",
        )

    try:
        logger.info("Research request received")

        result = run_research(payload.question)

        logger.info("Research completed successfully")

        return {
            "question": payload.question,
            "result": result,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Research pipeline failed")
        raise HTTPException(
            status_code=502,
            detail="Research failed. Please try again later.",
        )
