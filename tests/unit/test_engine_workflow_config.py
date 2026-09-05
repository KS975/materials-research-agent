from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.config import Settings


class EngineWorkflowConfigTests(unittest.TestCase):
    def test_model_status_policy_accepts_delivery_statuses(self) -> None:
        settings = Settings(
            _env_file=None,
            engine_allowed_model_statuses="approved, active",
        )
        self.assertEqual(settings.engine_allowed_model_statuses, "approved, active")

    def test_model_status_policy_rejects_unknown_or_empty_statuses(self) -> None:
        for value in ("", "ACTIVE,DEPRECATED"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Settings(_env_file=None, engine_allowed_model_statuses=value)

    def test_engine_task_settings_accept_delivery_ranges(self) -> None:
        settings = Settings(
            _env_file=None,
            engine_task_workers=4,
            engine_task_max_result_chars=3000000,
            engine_task_checkpoint_retries=2,
            engine_task_lease_seconds=120,
            engine_task_max_events=100,
        )
        self.assertEqual(settings.engine_task_workers, 4)
        self.assertEqual(settings.engine_task_max_events, 100)

    def test_engine_task_settings_reject_invalid_ranges(self) -> None:
        for field, value in (
            ("engine_task_workers", 0),
            ("engine_task_max_result_chars", 99999),
            ("engine_task_checkpoint_retries", 0),
            ("engine_task_lease_seconds", 1),
            ("engine_task_max_events", 1),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                Settings(_env_file=None, **{field: value})


if __name__ == "__main__":
    unittest.main()
