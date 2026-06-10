# ama2/backend/app/main.py

from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI

    HAS_FASTAPI = True
except Exception:  # pragma: no cover - optional dependency fallback
    HAS_FASTAPI = False

    class FastAPI:  # type: ignore[override]
        def __init__(self, title: str, version: str, lifespan=None):
            self.title = title
            self.version = version
            self.lifespan = lifespan
            self.routes: dict[str, Any] = {}
            self.state = type("state", (), {})()

        def get(self, path: str):
            def decorator(handler):
                self.routes[path] = handler
                return handler

            return decorator

from .config import settings
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI."""
    setup_logging(settings.LOG_LEVEL)
    logger.info("ama2_api_startup", environment=settings.environment)
    yield
    logger.info("ama2_api_shutdown", environment=settings.environment)

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
