from fastapi import APIRouter, Depends

from app.container import ApplicationContainer, get_container

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "materials-research-agent",
        "version": "0.1.1-dev1",
    }


@router.get("/api/v1/health/business-db")
def business_db_health(
    container: ApplicationContainer = Depends(get_container),
):
    settings = container.settings
    configured = bool(
        settings.business_db_host.strip()
        and settings.business_db_user.strip()
        and settings.business_db_password.get_secret_value()
        and settings.business_db_name.strip()
    )
    response = {
        "configured": configured,
        "connected": False,
        "database": settings.business_db_name,
        "session_read_only": False,
        "error_class": None,
    }
    if not configured:
        response["error_class"] = "NotConfigured"
        return response

    try:
        probe = container.db.ping()
        response["connected"] = True
        response["session_read_only"] = (
            int(probe.get("session_read_only", 0)) == 1
        )
    except Exception as exc:
        response["error_class"] = type(exc).__name__
    return response
