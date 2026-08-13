from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from app.config import Settings
from file_processing import ChatFileParser
from knowledge.embeddings import OpenAICompatibleEmbeddingProvider
from knowledge.file_ingestion import KnowledgeFileIngestionService
from knowledge.repository import QdrantKnowledgeRepository
from schemas.user_context import UserContext


def _parse_project_ids(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="V0.1.2 T05 real file -> Qdrant acceptance")
    parser.add_argument("--file", required=True, help="PDF or DOCX path")
    parser.add_argument("--query", required=True, help="semantic query to verify retrieval")
    parser.add_argument("--project-id", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    settings.require_knowledge()

    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    project_ids = _parse_project_ids(os.getenv("DEV_PROJECT_IDS", ""))

    if not user_id or not company_id or not project_ids:
        print("ERROR: configure DEV_USER_ID, DEV_COMPANY_ID, DEV_PROJECT_IDS in .env")
        return 2

    project_id = args.project_id if args.project_id is not None else project_ids[0]
    ctx = UserContext(
        user_id=user_id,
        company_id=company_id,
        project_ids=project_ids,
        permission_source="development_header",
    )
    if not ctx.can_access_project(project_id):
        print(f"ERROR: project {project_id} is outside DEV_PROJECT_IDS={project_ids}")
        return 2

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"ERROR: file not found: {file_path}")
        return 2

    embedding = OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key_value(),
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=float(settings.embedding_timeout),
    )

    repo = None
    try:
        if settings.qdrant_mode == "local":
            repo = QdrantKnowledgeRepository.local(
                path=settings.qdrant_local_path,
                embedding_provider=embedding,
                collection_name=settings.qdrant_collection,
            )
        else:
            repo = QdrantKnowledgeRepository.server(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() or None,
                embedding_provider=embedding,
                collection_name=settings.qdrant_collection,
            )

        service = KnowledgeFileIngestionService(ChatFileParser())
        result = service.index_bytes(
            filename=file_path.name,
            content=file_path.read_bytes(),
            media_type="",
            project_id=project_id,
            ctx=ctx,
            repository=repo,
        )

        print("INDEXED")
        print("document_id:", result.document_id)
        print("source_id:", result.source_id)
        print("filename:", result.filename)
        print("parser:", result.parser)
        print("project_id:", result.project_id)
        print("chunks_indexed:", result.chunks_indexed)
        print("secrets_redacted:", result.secrets_redacted)

        hits = repo.search(
            query=args.query,
            company_id=ctx.company_id,
            project_ids=[project_id],
            limit=5,
        )

        print("hits:", len(hits))
        for index, hit in enumerate(hits, 1):
            print(
                f"{index}. score={hit.score:.6f} project={hit.chunk.project_id} "
                f"source_id={hit.chunk.source_id} file={hit.chunk.filename}"
            )
            location = (
                f"page={hit.chunk.page_number}"
                if hit.chunk.page_number is not None
                else (
                    f"paragraph={hit.chunk.paragraph_start}-{hit.chunk.paragraph_end}"
                    if hit.chunk.paragraph_start is not None
                    else f"chunk={hit.chunk.chunk_index}"
                )
            )
            print("   ", location)
            print("   ", hit.chunk.text[:300])

        if not hits:
            print("FAIL: indexed file could not be retrieved")
            return 1
        if any(hit.chunk.project_id != project_id for hit in hits):
            print("FAIL: project permission filter leaked another project")
            return 1
        if not any(hit.chunk.document_id == result.document_id for hit in hits):
            print("FAIL: indexed document was not returned by semantic search")
            return 1

        print("T05 REAL FILE KNOWLEDGE INDEX PASS")
        return 0
    finally:
        if repo is not None:
            repo.close()
        embedding.close()


if __name__ == "__main__":
    raise SystemExit(main())
