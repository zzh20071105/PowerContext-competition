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

"""Original visible-event capture for one selected Pi session branch.

The archive layer validates the whole JSONL file before returning anything. The
render layer turns each record on the selected ancestor chain into its literal
visible body following the challenge's fixed table. The connector layer imports
only events missing from its durable checkpoint and refuses rewritten history.
"""

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


# ---------------------------------------------------------------------------
# Archive layer: decode, validate and index one complete session file.
# ---------------------------------------------------------------------------


def _decode_archive(content: bytes) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decode strict UTF-8 (BOM allowed) JSONL and validate the record graph.

    Returns ``(header, records)`` where ``records`` maps each record id to its
    parsed object in file order. Any violation of the archive contract raises
    ``ValueError`` before anything is returned.
    """

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
            raise ValueError(message)  # noqa: TRY004 - archive contract failures are ValueError
        if header is None:
            _require_header(row)
            header = row
            continue
        _register_record(row, header, records)
    if header is None:
        message = "Pi session archive must start with a session header"
        raise ValueError(message)
    return header, records


def _require_header(row: dict[str, Any]) -> None:
    if row.get("type") != "session" or row.get("version") != 3 or not _is_text(row.get("id")):
        message = "first record must be a type=session version=3 header with a non-empty id"
        raise ValueError(message)


def _register_record(row: dict[str, Any], header: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    record_id = row.get("id")
    if not _is_text(record_id):
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


def _ancestor_chain(
    header: dict[str, Any],
    records: dict[str, dict[str, Any]],
    leaf_id: str | None,
) -> list[dict[str, Any]]:
    """Return the selected leaf's ancestor chain ordered root first.

    The chain terminates at a record without a parent; a parent equal to the
    session header id is a legal root link because the header has appeared
    before every record. Passing ``None`` selects the file's last record.
    """

    selected = leaf_id if leaf_id is not None else (next(reversed(records)) if records else None)
    if selected is None or selected not in records:
        message = "selected leaf is missing from the session"
        raise ValueError(message)
    chain: list[dict[str, Any]] = []
    current: dict[str, Any] | None = records[selected]
    while current is not None:
        chain.append(current)
        parent_id = current.get("parentId")
        current = records.get(parent_id) if isinstance(parent_id, str) and parent_id else None
    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Render layer: one record to its literal visible body, per the fixed table.
# ---------------------------------------------------------------------------


def _render_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    """Render one record, or ``None`` when the record has no visible body."""

    event_type = record.get("type")
    parent_id = record.get("parentId") or None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not _is_text(role):
            return None
        body = _message_body(role, message)
    elif event_type in _SUMMARY_EVENT_TYPES:
        role = event_type
        summary = record.get("summary")
        if not _is_text(summary):
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
        prefix = f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}]"
        # Original text is preserved byte for byte; the prefix loses its trailing
        # space only when there is no text at all.
        return f"{prefix} {text}" if text else prefix
    if role == "bashExecution":
        return f"$ {message.get('command', '')}\n{message.get('output', '')}"
    return _content_body(message.get("content"))


def _content_body(content: object) -> str:
    """Render message content: strings as-is, blocks one line each, joined by ``\\n``."""

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
            parts.append(
                f"[tool_call {block.get('name', '')} {block.get('id', '')}] {_compact_json(block.get('arguments'))}"
            )
    return "\n".join(parts)


def _text_body(content: object) -> str:
    """Join the visible string or text blocks of a content value with ``\\n``."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _compact_json(arguments: object) -> str:
    """Compact separators, sorted keys, Unicode preserved."""

    return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value != ""


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events on the selected leaf's full ancestor chain.

    The archive is decoded, validated and the chain resolved completely before
    any event is returned; encoding, JSON, duplicate-id, missing-parent, cycle
    and missing-leaf failures raise ``ValueError`` without partial results.
    """

    header, records = _decode_archive(value.content)
    chain = _ancestor_chain(header, records, value.leaf_id)
    return tuple(
        event
        for event in (_render_event(header["id"], record) for record in chain)
        if event is not None
    )


# ---------------------------------------------------------------------------
# Source definition: identity, projection and read round-trip.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Connector layer: incremental import guarded by the durable checkpoint.
# ---------------------------------------------------------------------------


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiSessionConnector:
    """Incrementally import the selected branch of one on-disk Pi session file.

    Each run re-parses the whole file, then submits only the selected branch's
    visible events missing from the checkpoint. The checkpoint carries the
    identity and record digest of every event imported so far, so restarts
    deduplicate and rewritten imported history is rejected with ``ValueError``
    before anything is submitted; a rejected run leaves the committed progress
    untouched and the caller can retry after repairing the input.
    """

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        header, records = _decode_session_file(self.path)
        previous = _Checkpoint.load(session.checkpoint)
        previous.require_same_session(header["id"])
        previous.require_unchanged(records)
        events = tuple(
            event
            for event in (_render_event(header["id"], record) for record in _ancestor_chain(header, records, self.leaf_id))
            if event is not None
        )
        checkpoint = previous.advance()
        if not checkpoint.session_id:
            checkpoint.session_id = header["id"]
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if event.event_id in checkpoint.records:
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            checkpoint.records[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint.dump())


def _decode_session_file(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        content = path.read_bytes()
    except OSError as error:
        message = "Pi session file could not be read"
        raise ValueError(message) from error
    return _decode_archive(content)


class _Checkpoint:
    """Durable progress: the session id plus a digest per imported event."""

    def __init__(self, session_id: str, records: dict[str, str]) -> None:
        self.session_id = session_id
        self.records = records

    @classmethod
    def load(cls, checkpoint: object) -> _Checkpoint:
        if not isinstance(checkpoint, dict):
            return cls("", {})
        session_id = checkpoint.get("session_id")
        records = checkpoint.get("records")
        return cls(
            session_id if isinstance(session_id, str) else "",
            {
                event_id: digest
                for event_id, digest in (records or {}).items()
                if isinstance(event_id, str) and isinstance(digest, str)
            },
        )

    def require_same_session(self, session_id: str) -> None:
        if self.session_id and session_id and self.session_id != session_id:
            message = "session id changed since the last checkpoint"
            raise ValueError(message)

    def require_unchanged(self, records: dict[str, dict[str, Any]]) -> None:
        for event_id, digest in self.records.items():
            current = records.get(event_id)
            if current is None:
                message = f"imported event {event_id!r} is missing from the session file"
                raise ValueError(message)
            if _record_digest(current) != digest:
                message = f"imported event {event_id!r} was rewritten"
                raise ValueError(message)

    def advance(self) -> _Checkpoint:
        return _Checkpoint(self.session_id, dict(self.records))

    def dump(self) -> dict[str, object]:
        return {"session_id": self.session_id, "records": dict(self.records)}


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
