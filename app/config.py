from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "materials-research-agent"
    app_env: str = "development"
    log_level: str = "INFO"

    business_db_host: str = "127.0.0.1"
    business_db_port: int = 3306
    business_db_user: str = ""
    business_db_password: SecretStr = SecretStr("")
    business_db_name: str = "materials"
    business_db_charset: str = "utf8mb4"
    business_db_connect_timeout: int = 10
    business_db_read_timeout: int = 30

    permission_mode: Literal["development_header", "platform"] = "development_header"

    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout: int = 60

    # V0.1.2-A: current Chat temporary attachments
    chat_upload_dir: str = ".runtime/chat_uploads"
    chat_upload_max_mb: int = 25
    chat_upload_ttl_minutes: int = 180

    # V0.1.2-B: long-term Knowledge Index / Qdrant
    knowledge_upload_max_mb: int = 50
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    dashscope_api_key: SecretStr = SecretStr("")
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 10
    embedding_timeout: int = 60

    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: str = ".runtime/qdrant_knowledge"
    qdrant_collection: str = "materials_knowledge_v012"
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")

    # V0.1.2 T06: historical RAG retrieval guardrails
    knowledge_rag_score_threshold: float = 0.42
    knowledge_rag_max_hits: int = 5

    runtime_enabled: bool = False

    runtime_db_host: str = ""
    runtime_db_port: int = 3306
    runtime_db_user: str = ""
    runtime_db_password: SecretStr = SecretStr("")
    runtime_db_name: str = "materials_agent_runtime"
    runtime_db_charset: str = "utf8mb4"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_runtime_is_separate(self) -> "Settings":
        if self.runtime_enabled:
            if not self.runtime_db_name.strip():
                raise ValueError("RUNTIME_DB_NAME 不能为空")
            if self.runtime_db_name.strip().lower() == self.business_db_name.strip().lower():
                raise ValueError(
                    "Agent Runtime 必须使用独立数据库；RUNTIME_DB_NAME 不能等于 BUSINESS_DB_NAME"
                )
        return self


    def embedding_api_key_value(self) -> str:
        return (
            self.embedding_api_key.get_secret_value().strip()
            or self.dashscope_api_key.get_secret_value().strip()
        )

    def require_knowledge(self) -> None:
        missing = []
        if not self.embedding_base_url.strip():
            missing.append("EMBEDDING_BASE_URL")
        if not self.embedding_api_key_value():
            missing.append("EMBEDDING_API_KEY 或 DASHSCOPE_API_KEY")
        if not self.embedding_model.strip():
            missing.append("EMBEDDING_MODEL")
        if self.embedding_dimension <= 0:
            missing.append("EMBEDDING_DIMENSION")
        if self.qdrant_mode == "server" and not self.qdrant_url.strip():
            missing.append("QDRANT_URL")
        if missing:
            raise RuntimeError("缺少 Knowledge/Qdrant 配置：" + ", ".join(missing))

    def require_business_db(self) -> None:
        missing = []
        if not self.business_db_host.strip():
            missing.append("BUSINESS_DB_HOST")
        if not self.business_db_user.strip():
            missing.append("BUSINESS_DB_USER")
        if not self.business_db_password.get_secret_value():
            missing.append("BUSINESS_DB_PASSWORD")
        if not self.business_db_name.strip():
            missing.append("BUSINESS_DB_NAME")
        if missing:
            raise RuntimeError("缺少业务数据库配置：" + ", ".join(missing))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
