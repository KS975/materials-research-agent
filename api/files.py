from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from file_processing import EmptyDocumentError, UnsupportedFileTypeError
from schemas.user_context import UserContext

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.post("/chat-upload")
async def upload_chat_attachment(
    file: UploadFile = File(...),
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    content = await file.read()
    max_bytes = container.settings.chat_upload_max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"当前 Chat 单文件不能超过 {container.settings.chat_upload_max_mb} MB",
        )

    try:
        parsed = container.chat_file_parser.parse_bytes(
            filename,
            content,
            file.content_type or "",
        )
        attachment = container.chat_attachment_store.save(
            ctx=ctx,
            filename=filename,
            media_type=file.content_type or parsed.media_type,
            original_bytes=content,
            parsed=parsed,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"附件解析失败：{type(exc).__name__}: {exc}",
        ) from exc

    return {
        "status": "ok",
        "attachment_id": attachment.attachment_id,
        "filename": attachment.filename,
        "parser": attachment.parser,
        "page_count": attachment.page_count,
        "char_count": attachment.char_count,
        "chunk_count": attachment.chunk_count,
        "created_at": attachment.created_at,
        "expires_at": attachment.expires_at,
        "temporary": True,
        "indexed": False,
    }


@router.delete("/chat-attachments/{attachment_id}")
def delete_chat_attachment(
    attachment_id: str,
    ctx: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    try:
        container.chat_attachment_store.delete(attachment_id, ctx)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"status": "deleted", "attachment_id": attachment_id}
