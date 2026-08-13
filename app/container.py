from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

from agent.core import AgentCore
from agent.service import MaterialsAgentService
from agent.tool_registry import ToolRegistry
from agent.tools import MaterialsTools
from app.config import Settings, get_settings
from data.dynamic_fields import DynamicFieldResolver
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories import (
    ArchiveRepository,
    ColumnDefinitionRepository,
    ExperimentRepository,
    MaterialRepository,
    ProjectRepository,
    SampleRepository,
)
from llm.factory import create_llm_provider
from file_processing import UnifiedFileParser
from runtime.chat_attachments import ChatAttachmentStore
from skills.current_attachment import CurrentAttachmentSkill
from skills.historical_knowledge import HistoricalKnowledgeRAGSkill
from skills.joint_mysql_knowledge import JointMySQLKnowledgeAnalysisSkill
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

        self.registry = ToolRegistry()
        self.registry.register(
            "get_sample_context",
            "读取样品完整研发上下文：项目、配方、工艺、性能、测试条件和可选实验记录",
            self.tools.get_sample_context,
        )
        self.registry.register("get_formula", "读取样品配方", self.tools.get_formula)
        self.registry.register("get_process", "读取样品工艺", self.tools.get_process)
        self.registry.register("get_performance", "读取样品性能及测试条件", self.tools.get_performance)
        self.registry.register("compare_samples", "比较两个样品的配方、工艺、性能和测试条件", self.tools.compare_samples)
        self.registry.register("find_samples", "在当前公司/项目权限范围内查找样品", self.tools.find_samples)

        self.llm = create_llm_provider(settings)
        # V0.1.2-A: Current Chat temporary attachments
        self.file_parser = UnifiedFileParser()
        # Compatibility name kept for the frozen V0.1.2-A upload endpoint.
        self.chat_file_parser = self.file_parser

        self.chat_attachment_store = ChatAttachmentStore(
            settings.chat_upload_dir,
            settings.chat_upload_ttl_minutes,
        )

        self.current_attachment_skill = CurrentAttachmentSkill(
            self.chat_attachment_store,
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

        # V0.1.2 T07: read-only MySQL facts + same-project historical RAG.
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
        )
        self.runtime = create_runtime_store(settings)
        self.agent = MaterialsAgentService(self.core, self.runtime)

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
