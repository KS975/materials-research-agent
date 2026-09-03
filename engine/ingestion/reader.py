from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engine.contracts import FieldMetadata, FieldRole
from engine.exceptions import ValidationError


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    source_uri: str
    source_hash: str
    field_metadata: list[FieldMetadata]
    warnings: list[str]

    @property
    def fields(self) -> list[str]:
        return list(self.dataframe.columns)


def ingest_dataframe(
    dataframe: pd.DataFrame,
    *,
    source_uri: str = "dataframe",
    identifier_fields: list[str] | None = None,
    target_fields: list[str] | None = None,
    ignored_fields: list[str] | None = None,
) -> IngestionResult:
    if dataframe is None or not isinstance(dataframe, pd.DataFrame):
        raise ValidationError("input must be a pandas DataFrame")
    if dataframe.columns.has_duplicates:
        raise ValidationError("input columns must be unique")
    if dataframe.empty:
        raise ValidationError("input DataFrame is empty")

    identifiers = set(identifier_fields or [])
    targets = set(target_fields or [])
    ignored = set(ignored_fields or [])
    missing_columns = (identifiers | targets | ignored) - set(dataframe.columns)
    if missing_columns:
        raise ValidationError(f"configured fields missing from DataFrame: {sorted(missing_columns)}")

    metadata: list[FieldMetadata] = []
    for column in dataframe.columns:
        if column in identifiers:
            role = FieldRole.identifier
        elif column in targets:
            role = FieldRole.target
        elif column in ignored:
            role = FieldRole.ignored
        else:
            role = FieldRole.feature
        metadata.append(FieldMetadata(name=str(column), role=role, dtype=str(dataframe[column].dtype)))

    return IngestionResult(
        dataframe=dataframe.copy(deep=True),
        source_uri=source_uri,
        source_hash=hash_dataframe(dataframe),
        field_metadata=metadata,
        warnings=[],
    )


def read_tabular(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise ValidationError(f"input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise ValidationError(f"unsupported tabular input: {suffix}")


def hash_dataframe(dataframe: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(map(str, dataframe.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
    return digest.hexdigest()
