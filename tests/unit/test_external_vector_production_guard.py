from __future__ import annotations

import pytest
from app.config import Settings


def test_production_rejects_external_vector_test_company_override():
    with pytest.raises(ValueError, match="EXTERNAL_VECTOR_QUERY_COMPANY_ID"):
        Settings(
            _env_file=None,
            app_env="production",
            vector_search_provider="external_api",
            external_vector_query_company_id="test-company-id",
        )


def test_production_allows_empty_external_vector_override():
    settings = Settings(
        _env_file=None,
        app_env="production",
        vector_search_provider="external_api",
        external_vector_query_company_id="",
    )
    assert settings.external_vector_query_company_id == ""
