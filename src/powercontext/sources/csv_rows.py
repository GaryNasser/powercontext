# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture offline CSV records as one exact, citable Source per row."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """One complete CSV document supplied by an offline caller."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated record with its original header order and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured row evidence whose identity is derived from document and key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one UTF-8 CSV document before returning any row."""

    document_id = _require_document_id(value.document_id)
    if not value.primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV document must contain a header record") from None  # noqa: TRY003
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error  # noqa: TRY003

    _validate_header(header)
    if value.primary_key not in header:
        raise ValueError("primary_key must name an exact header column")  # noqa: TRY003
    return _parse_data_rows(reader, document_id, value.primary_key, tuple(header))


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable, delimiter-safe identity for one CSV row."""

    document = _require_document_id(document_id)
    key = _require_primary_key_value(primary_key_value)
    source_id = f"csv:{quote(document, safe='')}:{quote(key, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"CSV row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters"
        )
    return source_id


class CSVRowSourceAdapter:
    """Resolve row inputs into captured Sources and read them back exactly."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as standard text evidence in original column order."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)


def _parse_data_rows(
    reader: Iterator[list[str]],
    document_id: str,
    primary_key: str,
    columns: tuple[str, ...],
) -> tuple[CSVRowInput, ...]:
    key_index = columns.index(primary_key)
    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    try:
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV record width must match the header width")  # noqa: TRY003
            primary_key_value = _require_primary_key_value(record[key_index])
            if primary_key_value in seen_keys:
                raise ValueError("primary_key values must be unique within a document")  # noqa: TRY003
            seen_keys.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=document_id,
                    primary_key=primary_key,
                    primary_key_value=primary_key_value,
                    columns=columns,
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error  # noqa: TRY003
    return tuple(rows)


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header must contain at least one column")  # noqa: TRY003
    if any(not column.strip() for column in header):
        raise ValueError("CSV header columns must be non-empty and non-blank")  # noqa: TRY003
    if len(set(header)) != len(header):
        raise ValueError("CSV header columns must be unique")  # noqa: TRY003


def _require_document_id(document_id: str) -> str:
    if not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain surrounding whitespace")  # noqa: TRY003
    return document_id


def _require_primary_key_value(primary_key_value: str) -> str:
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty string")  # noqa: TRY003
    return primary_key_value


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
