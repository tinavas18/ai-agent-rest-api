
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import run_research

load_dotenv()

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


class ResearchRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=1000,
        description="The topic or question to research",
    )


@app.get("/")
def home():
    return {
        "message": "AI Research Agent API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/research")
@limiter.limit("5/minute")
def research(request: Request, payload: ResearchRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured.",
        )

    try:
        result = run_research(payload.question)
        return {
            "question": payload.question,
            "result": result,
        }
    except Exception:
        # Avoid exposing API keys or internal errors.
        raise HTTPException(
            status_code=502,
            detail="Research failed. Please try again later.",
        )
