import pytest

from data.mysql.client import ReadOnlyViolation, assert_read_only_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM eln_sample WHERE id=%s",
        "SHOW TABLES",
        "DESCRIBE eln_sample",
        "EXPLAIN SELECT * FROM eln_sample",
        "WITH x AS (SELECT 1) SELECT * FROM x",
    ],
)
def test_read_only_sql_allows_reads(sql):
    assert_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE eln_sample SET name='x'",
        "DELETE FROM eln_sample",
        "INSERT INTO eln_sample(name) VALUES ('x')",
        "DROP TABLE eln_sample",
        "SELECT 1; DELETE FROM eln_sample",
        "WITH x AS (SELECT 1) UPDATE eln_sample SET name='x'",
    ],
)
def test_read_only_sql_rejects_writes(sql):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only_sql(sql)
