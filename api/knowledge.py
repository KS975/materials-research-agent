from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from file_processing import EmptyDocumentError, UnsupportedFileTypeError
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # Optional: omitted means the caller's full authorized project scope.
    project_id: int | None = None
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/index-upload")
async def index_knowledge_upload(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    source_id: str | None = Form(default=None),
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    if not ctx.can_access_project(project_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前用户无权将文件写入该项目知识范围",
        )

    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    content = await file.read()
    max_bytes = container.settings.knowledge_upload_max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"知识索引单文件不能超过 "
                f"{container.settings.knowledge_upload_max_mb} MB"
            ),
        )

    try:
        with container.open_knowledge_repository() as repo:
            result = container.knowledge_file_ingestion.index_bytes(
                filename=filename,
                content=content,
                media_type=file.content_type or "",
                project_id=project_id,
                ctx=ctx,
                repository=repo,
                source_id=source_id,
            )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"知识索引失败：{type(exc).__name__}: {exc}",
        ) from exc

    return {
        "status": "indexed",
        "persistent": True,
        "temporary": False,
        "qdrant_mode": container.settings.qdrant_mode,
        "collection": container.settings.qdrant_collection,
        "document_id": result.document_id,
        "source_id": result.source_id,
        "filename": result.filename,
        "parser": result.parser,
        "page_count": result.page_count,
        "char_count": result.char_count,
        "chunks_indexed": result.chunks_indexed,
        "content_hash": result.content_hash,
        "company_id": result.company_id,
        "project_id": result.project_id,
    }


@router.post("/search")
def search_knowledge(
    body: KnowledgeSearchRequest,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    if body.project_id is not None:
        if not ctx.can_access_project(body.project_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="当前用户无权检索该项目知识范围",
            )
        search_project_ids = [int(body.project_id)]
        search_all_projects = False
        scope = {
            "mode": "explicit_project",
            "company_id": ctx.company_id,
            "project_ids": search_project_ids,
        }
    elif ctx.all_projects:
        search_project_ids = []
        search_all_projects = True
        scope = {
            "mode": "company_all_projects",
            "company_id": ctx.company_id,
            "project_ids": "*",
        }
    elif ctx.project_ids:
        search_project_ids = sorted({int(item) for item in ctx.project_ids})
        search_all_projects = False
        scope = {
            "mode": "authorized_projects",
            "company_id": ctx.company_id,
            "project_ids": search_project_ids,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前用户没有可用于历史知识检索的项目权限",
        )

    try:
        with container.open_knowledge_repository() as repo:
            hits = repo.search(
                query=body.query,
                company_id=ctx.company_id,
                project_ids=search_project_ids,
                all_projects=search_all_projects,
                limit=body.limit,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"知识检索失败：{type(exc).__name__}: {exc}",
        ) from exc

    items: list[dict[str, Any]] = []
    for hit in hits:
        chunk = hit.chunk
        items.append(
            {
                "score": hit.score,
                "document_id": chunk.document_id,
                "source_id": chunk.source_id,
                "filename": chunk.filename,
                "project_id": chunk.project_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "paragraph_start": chunk.paragraph_start,
                "paragraph_end": chunk.paragraph_end,
                "locator_type": chunk.locator_type,
            }
        )

    return {
        "status": "ok",
        "query": body.query,
        "company_id": ctx.company_id,
        "project_id": body.project_id,
        "retrieval_scope": scope,
        "count": len(items),
        "hits": items,
    }
