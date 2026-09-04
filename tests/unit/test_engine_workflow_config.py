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


if __name__ == "__main__":
    unittest.main()
