from __future__ import annotations

import os

import pytest

from agent.tools import MaterialsTools
from app.config import get_settings
from data.dynamic_fields import DynamicFieldResolver
from data.mysql.client import BusinessMySQLClient
from data.mysql.repositories import (
    ArchiveRepository,
    ColumnDefinitionRepository,
    ExperimentRepository,
    MaterialRepository,
    ProjectRepository,
    SampleRepository,
)
from schemas.user_context import UserContext


pytestmark = pytest.mark.integration


def _enabled():
    return os.getenv("RUN_REAL_MYSQL_TESTS") == "1"


@pytest.mark.skipif(not _enabled(), reason="Set RUN_REAL_MYSQL_TESTS=1 to run real DB tests")
def test_real_sample_context():
    company = os.environ["TEST_COMPANY_ID"]
    project_id = int(os.environ["TEST_PROJECT_ID"])
    sample_name = os.environ["TEST_SAMPLE_NAME"]
    user_id = os.getenv("TEST_USER_ID", "integration-test")

    settings = get_settings()
    db = BusinessMySQLClient(settings)
    samples = SampleRepository(db)
    tools = MaterialsTools(
        samples,
        ProjectRepository(db),
        ArchiveRepository(db),
        ExperimentRepository(db),
        DynamicFieldResolver(
            MaterialRepository(db),
            ColumnDefinitionRepository(db),
        ),
    )
    ctx = UserContext(
        user_id=user_id,
        company_id=company,
        project_ids=(project_id,),
        permission_source="integration_test",
    )

    result = tools.get_sample_context(sample_name, ctx)
    assert result["status"] == "ok"
    assert result["sample"]["name"] == sample_name
    assert result["sample"]["project_id"] == project_id
    assert result["evidence"]
