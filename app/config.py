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
    # ``platform`` mode accepts identity/scope only from an upstream gateway
    # that has already authenticated the Bearer token. The adapter decodes the
    # JWT payload solely to obtain a stable user identifier; it does not verify
    # the signature itself, so production must opt in explicitly.
    platform_trust_forwarded_headers: bool = False
    platform_jwt_user_claims: str = "userId,user_id,sub,id"

    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout: int = 60

    # DB Explorer V0.1. ``local_full`` sends authorized query results back to
    # the configured LLM, so it requires an explicit local-trust acknowledgement.
    database_explorer_mode: Literal["off", "schema_only", "local_full"] = "off"
    database_explorer_trust_local_llm: bool = False
    database_explorer_max_attempts: int = 4
    database_explorer_max_rows: int = 200
    database_explorer_query_timeout_ms: int = 8000
    database_explorer_max_result_chars: int = 60000

    # V0.1.2-A: current Chat temporary attachments
    chat_upload_dir: str = ".runtime/chat_uploads"
    chat_upload_max_mb: int = 25
    chat_upload_ttl_minutes: int = 180

    # LangGraph V3: durable, scoped Chat UI workflow checkpoints.
    chat_ui_workflow_dir: str = ".runtime/chat_ui_workflows"
    chat_ui_workflow_max_response_chars: int = 2_000_000
    chat_ui_workflow_checkpoint_retries: int = 3
    chat_ui_workflow_lease_seconds: int = 120
    chat_ui_workflow_ttl_hours: int = 72

    # Chat History V0.1: durable JSON conversations scoped by user + company.
    # In Docker, mount this directory to the host if history must survive
    # container recreation.
    chat_history_dir: str = ".runtime/chat_history"
    chat_history_max_messages: int = 400

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
        if self.database_explorer_mode != "off" and not self.llm_enabled:
            raise ValueError("启用 Database Explorer 时必须设置 LLM_ENABLED=true")
        if (
            self.database_explorer_mode == "local_full"
            and not self.database_explorer_trust_local_llm
        ):
            raise ValueError(
                "DATABASE_EXPLORER_MODE=local_full 时必须显式设置 "
                "DATABASE_EXPLORER_TRUST_LOCAL_LLM=true"
            )
        if not 1 <= self.database_explorer_max_attempts <= 8:
            raise ValueError("DATABASE_EXPLORER_MAX_ATTEMPTS 必须在 1 到 8 之间")
        if not 1 <= self.database_explorer_max_rows <= 1000:
            raise ValueError("DATABASE_EXPLORER_MAX_ROWS 必须在 1 到 1000 之间")
        if not 100 <= self.database_explorer_query_timeout_ms <= 120000:
            raise ValueError(
                "DATABASE_EXPLORER_QUERY_TIMEOUT_MS 必须在 100 到 120000 之间"
            )
        if not 100_000 <= self.chat_ui_workflow_max_response_chars <= 10_000_000:
            raise ValueError(
                "CHAT_UI_WORKFLOW_MAX_RESPONSE_CHARS 必须在100000到10000000之间"
            )
        if not 1 <= self.chat_ui_workflow_checkpoint_retries <= 5:
            raise ValueError(
                "CHAT_UI_WORKFLOW_CHECKPOINT_RETRIES 必须在1到5之间"
            )
        if not 10 <= self.chat_ui_workflow_lease_seconds <= 3600:
            raise ValueError(
                "CHAT_UI_WORKFLOW_LEASE_SECONDS 必须在10到3600之间"
            )
        if not 1 <= self.chat_ui_workflow_ttl_hours <= 720:
            raise ValueError(
                "CHAT_UI_WORKFLOW_TTL_HOURS 必须在1到720之间"
            )
        if not 20 <= self.chat_history_max_messages <= 2000:
            raise ValueError("CHAT_HISTORY_MAX_MESSAGES 必须在20到2000之间")
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
