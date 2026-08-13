from __future__ import annotations

import argparse
from dotenv import load_dotenv

from app.config import Settings
from knowledge.embeddings import OpenAICompatibleEmbeddingProvider
from knowledge.repository import QdrantKnowledgeRepository


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Delete one knowledge document from Qdrant by explicit scope."
    )
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--document-id", required=True)
    args = parser.parse_args()

    settings = Settings()
    settings.require_knowledge()

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

        repo.delete_document(
            company_id=args.company_id,
            project_id=args.project_id,
            document_id=args.document_id,
        )
        print("PURGED")
        print("company_id:", args.company_id)
        print("project_id:", args.project_id)
        print("document_id:", args.document_id)
        return 0
    finally:
        if repo is not None:
            repo.close()
        embedding.close()


if __name__ == "__main__":
    raise SystemExit(main())
