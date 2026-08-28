from fastapi import FastAPI

from api.chat import router as chat_router
from api.company_data import router as company_data_router
from api.health import router as health_router
from api.files import router as files_router
from api.feedback_ui import router as feedback_ui_router
from api.knowledge import router as knowledge_router
from api.ml_ui import router as ml_ui_router
from api.optimization_ui import router as optimization_ui_router
from api.autonomy_ui import router as autonomy_ui_router
from api.demo_ui import router as demo_ui_router
from api.dashboard import router as dashboard_router
from app.config import get_settings
from app.logging_config import configure_logging
from api.chat_ui import router as chat_ui_router


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Materials Research Agent",
    version="0.3",
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(chat_ui_router)
app.include_router(company_data_router)
app.include_router(ml_ui_router)
app.include_router(optimization_ui_router)
app.include_router(feedback_ui_router)
app.include_router(autonomy_ui_router)
app.include_router(demo_ui_router)
app.include_router(files_router)
app.include_router(knowledge_router)
app.include_router(dashboard_router)
