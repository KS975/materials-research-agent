from data.mysql.repositories.sample_repository import SampleRepository
from schemas.user_context import UserContext


class FakeDB:
    def __init__(self):
        self.sql = ""
        self.params = []

    def query_one(self, sql, params):
        self.sql = sql
        self.params = list(params)
        return {"id": 3073, "name": "ABS-051", "project_id": -1606}


def test_sample_lookup_all_company_projects_keeps_company_filter_without_project_whitelist():
    db = FakeDB()
    repo = SampleRepository(db)
    ctx = UserContext(
        user_id="u1",
        company_id="company-1",
        project_ids=(),
        permission_source="test",
        all_projects=True,
    )

    row = repo.get_by_id(3073, ctx)

    assert row["name"] == "ABS-051"
    assert "company = %s" in db.sql
    assert "1 = 1" in db.sql
    assert db.params == [3073, "company-1"]
