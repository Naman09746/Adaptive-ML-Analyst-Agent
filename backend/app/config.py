# ama2/backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ama2"
    REDIS_URL: str = "redis://localhost:6379/0"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    SECRET_KEY: str = "dev_secret_key"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
