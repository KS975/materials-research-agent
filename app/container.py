from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

from agent.core import AgentCore
from agent.engine_tool_registration import register_engine_tools
from agent.engine_workflow_adapter import EngineWorkflowAdapter
from agent.material_tool_registration import register_material_tools
from agent.service import MaterialsAgentService
from agent.scenario_composer import ScenarioWorkflowComposer
from agent.tool_registry import ToolRegistry
from agent.tools import MaterialsTools
from app.config import Settings, get_settings
from data.dynamic_fields import DynamicFieldResolver
from data.mysql.client import BusinessMySQLClient
from data.mysql.explorer import AuthorizedDatabaseExplorer
from data.mysql.repositories import (
    ArchiveRepository,
    ColumnDefinitionRepository,
    DashboardRepository,
    ExperimentRepository,
    MaterialRepository,
    ProjectRepository,
    SampleRepository,
)
from llm.factory import create_llm_provider
from file_processing import UnifiedFileParser
from runtime.chat_attachments import ChatAttachmentStore
from runtime.chat_ui_workflow import ChatUIWorkflowStore
from runtime.chat_history import ChatHistoryStore
from runtime.engine_tasks import EngineTaskManager, EngineTaskStore
from runtime.tool_audit import JsonlToolAuditStore
from skills.current_attachment import CurrentAttachmentSkill
from skills.database_explorer import DatabaseExplorerSkill
from skills.general_conversation import GeneralConversationFallbackSkill
from skills.historical_knowledge import HistoricalKnowledgeRAGSkill
from skills.joint_mysql_knowledge import JointMySQLKnowledgeAnalysisSkill
from skills.sample_historical_similarity import SampleHistoricalSimilaritySkill
from skills.catalog import build_default_skill_registry
from knowledge import OpenAICompatibleEmbeddingProvider, QdrantKnowledgeRepository
from knowledge.file_ingestion import KnowledgeFileIngestionService
from runtime.store import create_runtime_store


class ApplicationContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = BusinessMySQLClient(settings)

        self.samples = SampleRepository(self.db)
        self.projects = ProjectRepository(self.db)
        self.materials = MaterialRepository(self.db)
        self.columns = ColumnDefinitionRepository(self.db)
        self.dashboard = DashboardRepository(self.db)
        self.archives = ArchiveRepository(self.db)
        self.experiments = ExperimentRepository(self.db)

        self.resolver = DynamicFieldResolver(self.materials, self.columns)
        self.tools = MaterialsTools(
            samples=self.samples,
            projects=self.projects,
            archives=self.archives,
            experiments=self.experiments,
            resolver=self.resolver,
        )

        self.tool_audit_store = JsonlToolAuditStore(
            settings.tool_audit_dir,
            retries=settings.tool_audit_retries,
        )
        self.registry = ToolRegistry(
            audit_sink=(
                self.tool_audit_store.record
                if settings.tool_audit_enabled
                else None
            )
        )
        register_material_tools(self.registry, self.tools)
        register_engine_tools(self.registry)
        self.engine_workflow_adapter = EngineWorkflowAdapter(
            self.registry,
            artifact_root=settings.engine_artifact_root,
            enabled=settings.engine_workflow_enabled,
            max_source_rows=settings.engine_max_source_rows,
            default_algorithms=(
                item.strip()
                for item in settings.engine_default_algorithms.split(",")
            ),
            allowed_model_statuses=(
                item.strip()
                for item in settings.engine_allowed_model_statuses.split(",")
            ),
        )

        # Unified delivery architecture: fine-grained intents are compatibility
        # operation names.  Scenario Composer selects an atomic Skill contract,
        # whose Tool allow-list is enforced before any execution.
        self.skill_registry = build_default_skill_registry()
        self.scenario_composer = ScenarioWorkflowComposer(self.skill_registry)

        self.llm = create_llm_provider(settings)
        self.database_explorer = AuthorizedDatabaseExplorer(self.db)
        self.database_explorer_skill = DatabaseExplorerSkill(
            self.database_explorer,
            self.llm,
            mode=settings.database_explorer_mode,
            trust_local_llm=settings.database_explorer_trust_local_llm,
            max_attempts=settings.database_explorer_max_attempts,
            max_rows=settings.database_explorer_max_rows,
            query_timeout_ms=settings.database_explorer_query_timeout_ms,
            max_result_chars=settings.database_explorer_max_result_chars,
        )
        # V0.1.2-A: Current Chat temporary attachments
        self.file_parser = UnifiedFileParser()
        # Compatibility name kept for the frozen V0.1.2-A upload endpoint.
        self.chat_file_parser = self.file_parser

        self.chat_attachment_store = ChatAttachmentStore(
            settings.chat_upload_dir,
            settings.chat_upload_ttl_minutes,
        )
        self.chat_ui_workflow_store = ChatUIWorkflowStore(
            settings.chat_ui_workflow_dir,
            max_response_chars=settings.chat_ui_workflow_max_response_chars,
            checkpoint_retries=settings.chat_ui_workflow_checkpoint_retries,
            lease_seconds=settings.chat_ui_workflow_lease_seconds,
            ttl_hours=settings.chat_ui_workflow_ttl_hours,
        )
        self.chat_history_store = ChatHistoryStore(
            settings.chat_history_dir,
            max_messages=settings.chat_history_max_messages,
        )

        self.current_attachment_skill = CurrentAttachmentSkill(
            self.chat_attachment_store,
            self.llm,
        )
        self.general_conversation_skill = GeneralConversationFallbackSkill(
            self.llm,
        )

        # V0.1.2-B: long-term Knowledge Index reuses the same parser.
        self.knowledge_file_ingestion = KnowledgeFileIngestionService(
            self.file_parser
        )

        # V0.1.2 T06: historical Knowledge Index -> RAG.
        # The repository is opened per request so Qdrant Local file locks are
        # released immediately on Windows.
        self.historical_knowledge_skill = HistoricalKnowledgeRAGSkill(
            self.open_knowledge_repository,
            self.llm,
            score_threshold=settings.knowledge_rag_score_threshold,
            max_hits=settings.knowledge_rag_max_hits,
        )

        # One-sample MySQL facts + historical similarity RAG. Historical
        # project scope is independent from the sample's own project.
        self.sample_historical_similarity_skill = SampleHistoricalSimilaritySkill(
            self.registry,
            self.open_knowledge_repository,
            self.llm,
            score_threshold=settings.knowledge_rag_score_threshold,
            max_hits=settings.knowledge_rag_max_hits,
        )

        # V0.1.2 T07: read-only MySQL facts + historical RAG.
        self.joint_mysql_knowledge_skill = JointMySQLKnowledgeAnalysisSkill(
            self.registry,
            self.open_knowledge_repository,
            self.llm,
            score_threshold=settings.knowledge_rag_score_threshold,
            max_hits=settings.knowledge_rag_max_hits,
        )

        self.core = AgentCore(
            registry=self.registry,
            llm=self.llm,
            llm_enabled=settings.llm_enabled,
            skill_registry=self.skill_registry,
            scenario_composer=self.scenario_composer,
            engine_workflow_adapter=self.engine_workflow_adapter,
        )
        self.runtime = create_runtime_store(settings)
        self.agent = MaterialsAgentService(self.core, self.runtime)
        self.engine_task_store = EngineTaskStore(
            settings.engine_task_dir,
            max_result_chars=settings.engine_task_max_result_chars,
            checkpoint_retries=settings.engine_task_checkpoint_retries,
            lease_seconds=settings.engine_task_lease_seconds,
            max_events=settings.engine_task_max_events,
        )
        self.engine_task_manager = EngineTaskManager(
            core=self.core,
            store=self.engine_task_store,
            worker_count=settings.engine_task_workers,
        )
        self.engine_task_manager.recover_interrupted()

    @contextmanager
    def open_knowledge_repository(self):
        """Open one scoped Qdrant repository operation and close it cleanly.

        Local Mode is opened per operation so Windows file locks are released
        immediately after an index/search request. Server Mode uses the same
        upper-layer repository API.
        """
        self.settings.require_knowledge()

        embedding = OpenAICompatibleEmbeddingProvider(
            base_url=self.settings.embedding_base_url,
            api_key=self.settings.embedding_api_key_value(),
            model=self.settings.embedding_model,
            dimension=self.settings.embedding_dimension,
            batch_size=self.settings.embedding_batch_size,
            timeout_seconds=float(self.settings.embedding_timeout),
        )
        repo = None
        try:
            if self.settings.qdrant_mode == "local":
                repo = QdrantKnowledgeRepository.local(
                    path=self.settings.qdrant_local_path,
                    embedding_provider=embedding,
                    collection_name=self.settings.qdrant_collection,
                )
            else:
                repo = QdrantKnowledgeRepository.server(
                    url=self.settings.qdrant_url,
                    api_key=(
                        self.settings.qdrant_api_key.get_secret_value() or None
                    ),
                    embedding_provider=embedding,
                    collection_name=self.settings.qdrant_collection,
                )
            yield repo
        finally:
            if repo is not None:
                repo.close()
            embedding.close()


@lru_cache(maxsize=1)
def get_container() -> ApplicationContainer:
    return ApplicationContainer(get_settings())
