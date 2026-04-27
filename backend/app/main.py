# ama2/backend/app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from .utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI."""
    setup_logging()
    logger.info("ama2_api_startup")
    yield
    logger.info("ama2_api_shutdown")

def create_app() -> FastAPI:
    """FastAPI app factory."""
    app = FastAPI(
        title="AMA² — Adaptive ML Analyst Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ... (Register routers, middlewares etc. here)
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app

app = create_app()
