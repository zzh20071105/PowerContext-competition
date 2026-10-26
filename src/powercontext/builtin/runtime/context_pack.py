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

"""Chunked Context Packs with prefix-reuse deltas for incremental transport."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Protocol or chunking validation failure for Context Pack data."""


@dataclass(frozen=True)
class ContextPack:
    """One complete chunked context payload with digest and byte accounting."""

    items: tuple[bytes, ...] = ()

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        return cls(tuple(items))

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Chunk UTF-8 text deterministically into bounded valid-UTF-8 blocks."""

        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                message = "content bytes must be valid UTF-8"
                raise ContextPackError(message) from error
        else:
            text = content
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            message = "chunk_bytes must be a positive integer"
            raise ContextPackError(message)
        chunks: list[bytes] = []
        current = bytearray()
        for character in text:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                message = "no valid chunk boundary fits within chunk_bytes"
                raise ContextPackError(message)
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current = bytearray()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    @property
    def digest(self) -> str:
        return _digest_items(self.items)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload, _ = _encode_pack(self.items)
        return payload

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        document = _decode_json(raw)
        _require_exact_fields(document, _PACK_FIELDS)
        if document["schema"] != PACK_SCHEMA:
            message = "unexpected Context Pack schema"
            raise ContextPackError(message)
        items = _decode_items(document["items"])
        digest = _decode_digest(document["digest"])
        if digest != _digest_items(items):
            message = "Context Pack digest does not match its items"
            raise ContextPackError(message)
        byte_count = document["byte_count"]
        if type(byte_count) is not int or byte_count != len(raw):
            message = "Context Pack byte_count does not match its wire size"
            raise ContextPackError(message)
        return cls(items=items)


@dataclass(frozen=True)
class ContextPackDelta:
    """One prefix-reuse delta that rebuilds a target pack from a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = field(default=())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        maximum = min(len(base.items), len(target.items))
        while prefix < maximum and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _encode_delta(self.base_digest, self.target_digest, self.prefix_items, self.appended)

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        document = _decode_json(raw)
        _require_exact_fields(document, _DELTA_FIELDS)
        if document["schema"] != DELTA_SCHEMA:
            message = "unexpected Context Pack delta schema"
            raise ContextPackError(message)
        prefix_items = document["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            message = "prefix_items must be a non-negative integer"
            raise ContextPackError(message)
        return cls(
            base_digest=_decode_digest(document["base_digest"]),
            target_digest=_decode_digest(document["target_digest"]),
            prefix_items=prefix_items,
            appended=_decode_items(document["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from one base pack and one validated delta."""

    if delta.base_digest != base.digest:
        message = "delta base digest does not match the base pack"
        raise ContextPackError(message)
    if delta.prefix_items > len(base.items):
        message = "prefix_items exceeds the base pack chunk count"
        raise ContextPackError(message)
    rebuilt = ContextPack(base.items[: delta.prefix_items] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        message = "rebuilt pack digest does not match the delta target digest"
        raise ContextPackError(message)
    return rebuilt


def _digest_items(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [base64.b64encode(item).decode("ascii") for item in items]}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _encode_pack(items: tuple[bytes, ...]) -> tuple[bytes, int]:
    unsigned = {"schema": PACK_SCHEMA, "items": [base64.b64encode(item).decode("ascii") for item in items]}
    byte_count = 0
    while True:
        payload = {**unsigned, "digest": _digest_items(items), "byte_count": byte_count}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) == byte_count:
            return encoded, byte_count
        byte_count = len(encoded)


def _encode_delta(base_digest: str, target_digest: str, prefix_items: int, appended: tuple[bytes, ...]) -> bytes:
    payload = {
        "schema": DELTA_SCHEMA,
        "base_digest": base_digest,
        "target_digest": target_digest,
        "prefix_items": prefix_items,
        "appended": [base64.b64encode(item).decode("ascii") for item in appended],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _decode_json(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = "wire data must be UTF-8 encoded JSON"
        raise ContextPackError(message) from error
    if not isinstance(document, dict):
        message = "wire data must be a JSON object"
        raise ContextPackError(message)
    return document


def _require_exact_fields(document: dict[str, Any], fields: frozenset[str]) -> None:
    if set(document) != fields:
        message = "wire fields must exactly match the schema contract"
        raise ContextPackError(message)


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        message = "chunk arrays must be JSON lists"
        raise ContextPackError(message)
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            message = "chunks must be standard Base64 strings"
            raise ContextPackError(message)
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            message = "chunks must be standard Base64 strings"
            raise ContextPackError(message) from error
    return tuple(items)


def _decode_digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.match(value):
        message = "digests must be formatted as sha256:<lowercase hex>"
        raise ContextPackError(message)
    return value


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
