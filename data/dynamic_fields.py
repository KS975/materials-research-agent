from __future__ import annotations

import re
from typing import Any

from data.json_utils import decode_json_mapping
from data.mysql.repositories.column_repository import ColumnDefinitionRepository
from data.mysql.repositories.material_repository import MaterialRepository


_RECIPE_KEY = re.compile(r"^R3-(\d+)$")
_DYNAMIC_KEYS = {
    "performance": re.compile(r"^P(\d+)$"),
    "service_performance": re.compile(r"^SP(\d+)$"),
    "process": re.compile(r"^S(\d+)$"),
}


class DynamicFieldResolver:
    def __init__(
        self,
        materials: MaterialRepository,
        columns: ColumnDefinitionRepository,
    ):
        self.materials = materials
        self.columns = columns

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        return decode_json_mapping(value)

    def resolve_formula(self, raw: Any, company_id: str) -> list[dict[str, Any]]:
        mapping = self._as_mapping(raw)
        ids = []
        key_to_id: dict[str, int] = {}
        for key in mapping:
            match = _RECIPE_KEY.match(str(key))
            if match:
                field_id = int(match.group(1))
                ids.append(field_id)
                key_to_id[str(key)] = field_id

        definitions = self.materials.get_sample_materials(ids, company_id)
        result = []
        for key, value in mapping.items():
            key = str(key)
            field_id = key_to_id.get(key)
            definition = definitions.get(field_id) if field_id is not None else None
            result.append(
                {
                    "raw_key": key,
                    "field_id": field_id,
                    "name": definition.get("name") if definition else None,
                    "unit": definition.get("unit") if definition else None,
                    "value": value,
                    "resolved": bool(definition),
                    "source": "sample_materials" if definition else None,
                }
            )
        return result

    def resolve_dynamic(
        self,
        raw: Any,
        company_id: str,
        category: str,
    ) -> list[dict[str, Any]]:
        mapping = self._as_mapping(raw)
        pattern = _DYNAMIC_KEYS[category]
        ids = []
        key_to_id: dict[str, int] = {}
        for key in mapping:
            match = pattern.match(str(key))
            if match:
                field_id = int(match.group(1))
                ids.append(field_id)
                key_to_id[str(key)] = field_id

        definitions = self.columns.get_by_ids(ids, company_id)
        result = []
        for key, value in mapping.items():
            key = str(key)
            field_id = key_to_id.get(key)
            definition = definitions.get(field_id) if field_id is not None else None
            result.append(
                {
                    "raw_key": key,
                    "field_id": field_id,
                    "name": definition.get("name") if definition else None,
                    "unit": definition.get("unit") if definition else None,
                    "value": value,
                    "resolved": bool(definition),
                    "source": "data_column" if definition else None,
                }
            )
        return result
