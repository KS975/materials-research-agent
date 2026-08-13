from data.dynamic_fields import DynamicFieldResolver


class FakeMaterials:
    def get_sample_materials(self, ids, company_id):
        data = {
            401: {"id": 401, "name": "水", "unit": None},
            402: {"id": 402, "name": "P507+煤油", "unit": None},
        }
        return {x: data[x] for x in ids if x in data}


class FakeColumns:
    def get_by_ids(self, ids, company_id):
        data = {
            14598: {"id": 14598, "name": "密度差", "unit": None},
            14741: {"id": 14741, "name": "服役性能", "unit": None},
        }
        return {x: data[x] for x in ids if x in data}


def test_formula_mapping_uses_sample_materials():
    resolver = DynamicFieldResolver(FakeMaterials(), FakeColumns())
    rows = resolver.resolve_formula(
        {"R3-401": "81.1064", "R3-402": "16.0889"},
        "company",
    )
    assert rows[0]["name"] == "水"
    assert rows[0]["resolved"] is True
    assert rows[1]["name"] == "P507+煤油"


def test_performance_mapping_uses_data_column():
    resolver = DynamicFieldResolver(FakeMaterials(), FakeColumns())
    rows = resolver.resolve_dynamic(
        {"P14598": "41.2052"},
        "company",
        "performance",
    )
    assert rows == [
        {
            "raw_key": "P14598",
            "field_id": 14598,
            "name": "密度差",
            "unit": None,
            "value": "41.2052",
            "resolved": True,
            "source": "data_column",
        }
    ]


def test_unknown_field_is_not_invented():
    resolver = DynamicFieldResolver(FakeMaterials(), FakeColumns())
    rows = resolver.resolve_dynamic(
        {"P999999": "1"},
        "company",
        "performance",
    )
    assert rows[0]["resolved"] is False
    assert rows[0]["name"] is None
    assert rows[0]["raw_key"] == "P999999"



def test_mysql_json_string_is_decoded():
    resolver = DynamicFieldResolver(FakeMaterials(), FakeColumns())
    rows = resolver.resolve_formula(
        '{"R3-401":"81.1064","R3-402":"16.0889"}',
        "company",
    )
    assert [row["name"] for row in rows] == ["水", "P507+煤油"]
