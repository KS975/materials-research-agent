from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "materials-research-agent",
        "version": "0.1.1-dev1",
    }
