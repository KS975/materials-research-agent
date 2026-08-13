from fastapi import FastAPI

from api.chat import router as chat_router
from api.health import router as health_router
from api.files import router as files_router
from api.knowledge import router as knowledge_router
from api.ml_ui import router as ml_ui_router
from app.config import get_settings
from app.logging_config import configure_logging
from api.chat_ui import router as chat_ui_router


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Materials Research Agent",
    version="0.1.3-ui",
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(chat_ui_router)
app.include_router(ml_ui_router)
app.include_router(files_router)
app.include_router(knowledge_router)
