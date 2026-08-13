import pytest
from pydantic import ValidationError

from app.config import Settings


def test_runtime_cannot_use_business_database():
    with pytest.raises(ValidationError):
        Settings(
            business_db_name="materials",
            runtime_enabled=True,
            runtime_db_name="materials",
        )
