from __future__ import annotations

from typing import Any

from data.mysql.client import BusinessMySQLClient


class ExperimentRepository:
    """Optional evidence.

    Authorization is performed by fetching/authorizing the sample before calling these methods.
    The involved experiment tables do not expose company in the verified schema.
    """

    def __init__(self, db: BusinessMySQLClient):
        self.db = db

    def list_synthesis(self, sample_id: int) -> list[dict[str, Any]]:
        sql = """
            SELECT
                id, sample_id, note, attachments, state, reject_msg,
                synthesis_time, synthesis_user_id, synthesis_user,
                default_output, create_time, audit_time,
                audit_user, audit_user_id, craft
            FROM eln_synthesis_exp
            WHERE sample_id = %s
            ORDER BY id DESC
        """
        return self.db.query_all(sql, [int(sample_id)])

    def list_verify_items(self, sample_id: int) -> list[dict[str, Any]]:
        sql = """
            SELECT
                s.sample_id,
                s.verify_item_id,
                s.exp_id,
                i.name AS verify_item_name,
                i.conditions,
                i.performances,
                i.test_standard,
                i.project_id,
                i.describe,
                i.state
            FROM eln_verify_item_sample s
            LEFT JOIN eln_verify_item i
              ON s.verify_item_id = i.id
            WHERE s.sample_id = %s
            ORDER BY s.id DESC
        """
        return self.db.query_all(sql, [int(sample_id)])
