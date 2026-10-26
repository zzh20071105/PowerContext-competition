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

"""Row-level CSV Sources that keep original fields, column order and row identity."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
_MAX_SOURCE_ID_CHARS = MAX_SOURCE_ID_LENGTH


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document identified by a stable document identity."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One captured CSV record addressed by its document and primary-key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured original CSV row with its identity components and full record."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the fixed row identity ``csv:{document}:{key}`` with percent encoding."""

    _require_plain_component("document_id", document_id)
    _require_present_value("primary_key_value", primary_key_value)
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > _MAX_SOURCE_ID_CHARS:
        message = "csv row source_id must not exceed 256 characters"
        raise ValueError(message)
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one complete offline CSV document into validated row inputs.

    The whole document is decoded, parsed and validated before any row is returned;
    encoding, CSV-format and data-contract failures raise :class:`ValueError` without
    partial results.
    """

    _require_plain_component("document_id", value.document_id)
    if not isinstance(value.primary_key, str) or not value.primary_key:
        message = "primary_key must be a non-empty string"
        raise ValueError(message)
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        message = "CSV content must be strict UTF-8"
        raise ValueError(message) from error
    try:
        records = tuple(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        message = "CSV content is malformed"
        raise ValueError(message) from error
    if not records or not records[0]:
        message = "CSV document must start with a non-empty header record"
        raise ValueError(message)

    header = records[0]
    columns = tuple(header)
    _validate_header(columns, value.primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    key_index = header.index(value.primary_key)
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            message = "CSV data record width must match the header width"
            raise ValueError(message)
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            message = "primary key values must be non-empty and not whitespace-only"
            raise ValueError(message)
        if primary_key_value in seen_keys:
            message = "primary key values must be unique within one document"
            raise ValueError(message)
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowAdapter:
    """Resolve and read captured CSV rows under the ``csv-row`` Definition."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
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
    """Project one captured row as compact original-order JSON evidence."""

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
    CSVRowAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)


def _validate_header(columns: tuple[str, ...], primary_key: str) -> None:
    for column in columns:
        if not column.strip():
            message = "header fields must be non-empty and not whitespace-only"
            raise ValueError(message)
    if len(set(columns)) != len(columns):
        message = "header fields must be unique"
        raise ValueError(message)
    if primary_key not in columns:
        message = "primary_key must exactly match one header column"
        raise ValueError(message)


def _require_plain_component(field: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        message = f"{field} must be a non-empty string"
        raise ValueError(message)
    if value != value.strip():
        message = f"{field} must not contain leading or trailing whitespace"
        raise ValueError(message)


def _require_present_value(field: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        message = f"{field} must be a non-empty string that is not whitespace-only"
        raise ValueError(message)
