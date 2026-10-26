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

"""Original visible-event capture for one selected Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import (
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"

_MAX_SOURCE_ID_CHARS = MAX_SOURCE_ID_LENGTH
_SUMMARY_EVENT_TYPES = frozenset({"compaction", "branch_summary"})


class PiSessionInput(BaseModel):
    """One complete offline Pi v3 session archive with an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible session event with its complete original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue] = Field(default_factory=dict)


class PiEventSource(Source):
    """Captured original Pi session event kept as immutable evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue] = Field(default_factory=dict)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable event identity ``pi:{session}:{event}`` with percent encoding."""

    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > _MAX_SOURCE_ID_CHARS:
        message = "pi event source_id must not exceed 256 characters"
        raise ValueError(message)
    return source_id


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events on the selected leaf's full ancestor chain.

    The archive is decoded, parsed and graph-validated completely before anything is
    returned; encoding, JSON, duplicate-id, missing-parent, cycle and missing-leaf
    failures raise :class:`ValueError` without partial results.
    """

    header, records = _decode_session(value.content)
    return _visible_chain(header, records, value.leaf_id)


def _visible_chain(
    header: dict[str, Any],
    records: dict[str, dict[str, Any]],
    leaf_id: str | None,
) -> tuple[PiEventInput, ...]:
    selected = leaf_id if leaf_id is not None else (next(reversed(records)) if records else None)
    if selected is None or selected not in records:
        message = "selected leaf is missing from the session"
        raise ValueError(message)
    chain: list[dict[str, Any]] = []
    current: dict[str, Any] | None = records[selected]
    while current is not None:
        chain.append(current)
        parent_id = current.get("parentId")
        current = records[parent_id] if isinstance(parent_id, str) and parent_id else None
    return tuple(
        event
        for event in (_visible_event(header["id"], record) for record in reversed(chain))
        if event is not None
    )


def _decode_session(content: bytes) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        message = "Pi session content must be strict UTF-8"
        raise ValueError(message) from error
    header: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            message = "Pi session lines must be valid JSON"
            raise ValueError(message) from error
        if not isinstance(row, dict):
            message = "Pi session lines must be JSON objects"
            raise ValueError(message)
        if header is None:
            if row.get("type") != "session" or row.get("version") != 3 or not _is_non_empty_str(row.get("id")):
                message = "first record must be a type=session version=3 header with a non-empty id"
                raise ValueError(message)
            header = row
            continue
        record_id = row.get("id")
        if not _is_non_empty_str(record_id):
            message = "session records must carry a non-empty string id"
            raise ValueError(message)
        if record_id in records or record_id == header["id"]:
            message = "session record ids must be unique"
            raise ValueError(message)
        parent_id = row.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            message = "parentId must be null or a string"
            raise ValueError(message)
        if isinstance(parent_id, str) and parent_id and parent_id not in records and parent_id != header["id"]:
            message = "parentId must reference an earlier record"
            raise ValueError(message)
        records[record_id] = row
    if header is None:
        message = "Pi session archive must start with a session header"
        raise ValueError(message)
    return header, records


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    parent_id = record.get("parentId") or None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not _is_non_empty_str(role):
            return None
        body = _message_body(role, message)
    elif event_type in _SUMMARY_EVENT_TYPES:
        role = event_type
        summary = record.get("summary")
        if not _is_non_empty_str(summary):
            return None
        body = summary
    elif event_type == "custom_message":
        role = "custom"
        body = _text_body(record.get("content"))
    else:
        return None
    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str:
    if role == "toolResult":
        text = _text_body(message.get("content"))
        return f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {text}".rstrip()
    if role == "bashExecution":
        return f"$ {message.get('command', '')}\n{message.get('output', '')}"
    return _content_body(message.get("content"))


def _content_body(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            arguments = block.get("arguments")
            encoded = json.dumps(
                arguments if arguments is not None else {},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            parts.append(f"[tool_call {block.get('name', '')} {block.get('id', '')}] {encoded}")
    return "\n".join(parts)


def _text_body(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


class PiEventAdapter:
    """Resolve and read captured session events under the ``pi-event`` Definition."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Project one captured event as complete original text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Incrementally import the selected branch of one on-disk Pi session file."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        header, records = _decode_session(self.path.read_bytes())
        state = _CheckpointState.load(session.checkpoint)
        state.validate_unchanged(header, records)
        events = _visible_chain(header, records, self.leaf_id)
        imported = dict(state.imported)
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in imported:
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            imported[identity] = _digest_record(event.record)
        checkpoint = {
            "session_id": header["id"],
            "imported": imported,
            "digests": {record_id: _digest_record(record) for record_id, record in records.items()},
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


class _CheckpointState:
    """Opaque JSON checkpoint: imported identities plus digests of every file record."""

    def __init__(self, session_id: str, imported: dict[str, str], digests: dict[str, str]) -> None:
        self.session_id = session_id
        self.imported = imported
        self.digests = digests

    @classmethod
    def load(cls, checkpoint: object) -> _CheckpointState:
        if not isinstance(checkpoint, dict):
            return cls("", {}, {})
        session_id = checkpoint.get("session_id")
        imported = checkpoint.get("imported")
        digests = checkpoint.get("digests")
        return cls(
            session_id if isinstance(session_id, str) else "",
            {key: value for key, value in (imported or {}).items() if isinstance(key, str) and isinstance(value, str)},
            {key: value for key, value in (digests or {}).items() if isinstance(key, str) and isinstance(value, str)},
        )

    def validate_unchanged(self, header: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
        if not self.imported and not self.digests:
            return
        if self.session_id and header.get("id") != self.session_id:
            message = "session id changed since the last checkpoint"
            raise ValueError(message)
        for record_id, digest in self.digests.items():
            current = records.get(record_id)
            if current is None:
                message = f"previously imported record {record_id!r} is missing from the session file"
                raise ValueError(message)
            if _digest_record(current) != digest:
                message = f"previously imported record {record_id!r} was rewritten"
                raise ValueError(message)


def _digest_record(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
