# ama2/backend/app/config.py

from typing import Any, Literal

try:
    from pydantic import Field, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - optional dependency fallback
    HAS_PYDANTIC_SETTINGS = False

    class BaseSettings:  # type: ignore[override]
        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default: Any = None, **kwargs: Any):  # type: ignore[misc]
        return default

    def model_validator(*args: Any, **kwargs: Any):
        def decorator(function):
            return function

        return decorator

    class SettingsConfigDict(dict):
        pass


class Settings(BaseSettings):
    environment: Literal["development", "test", "production"] = "development"
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ama2",
        description="Async SQLAlchemy database URL.",
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis broker/cache URL.",
    )
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI.",
    )
    SECRET_KEY: str = Field(
        default="dev_secret_key",
        description="Application secret used for local development.",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=500)
    ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @model_validator(mode="after")
    def validate_security_settings(self):
        if self.environment == "production" and self.SECRET_KEY in {"dev_secret_key", "change-me", ""}:
            raise ValueError("SECRET_KEY must be set to a strong value in production")
        return self


settings = Settings()
