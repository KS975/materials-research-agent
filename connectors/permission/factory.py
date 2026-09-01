from app.config import Settings
from connectors.permission.development_header import DevelopmentHeaderPermissionAdapter
from connectors.permission.platform import PlatformPermissionAdapter


def create_permission_adapter(settings: Settings):
    if settings.permission_mode == "development_header":
        return DevelopmentHeaderPermissionAdapter()
    if settings.permission_mode == "platform":
        return PlatformPermissionAdapter(settings)
    raise RuntimeError(f"不支持的 PERMISSION_MODE: {settings.permission_mode}")
