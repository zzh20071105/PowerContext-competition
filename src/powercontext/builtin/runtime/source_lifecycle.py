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

"""Source withdrawal and restore propagation across Memory and Topic Memory evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, insert, select, update

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.errors import ArtifactNotFoundError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

KIND_SOURCE = "source"
KIND_MEMORY = "memory"
KIND_TOPIC = "topic"
_PURPOSES = frozenset({"generation", "audit"})


class SourceLifecycleError(ValueError):
    """One lifecycle registration or event violated the dependency contract."""


class SourceLifecycleEvent(BaseModel):
    """One durable availability change for exactly one registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_visible_text(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            message = f"{info.field_name} must be a non-empty string that is not whitespace-only"
            raise ValueError(message)
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def require_positive_integer(cls, value: int, info) -> int:
        if type(value) is not int or value < 1:
            message = f"{info.field_name} must be a positive integer"
            raise ValueError(message)
        return value


class SourceLifecycleApplication:
    """Select the scoped lifecycle service for one Runtime."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        return ScopedSourceLifecycleService(self._runtime, scope_id)


class _Node:
    __slots__ = ("available", "kind", "last_event", "last_sequence", "node_id", "parents", "policy", "ref", "revision")

    def __init__(
        self,
        node_id: str,
        kind: str,
        ref: dict[str, Any],
        parents: list[str],
        policy: str,
        revision: int,
        available: bool,
        last_sequence: int,
        last_event: dict[str, Any] | None,
    ) -> None:
        self.node_id = node_id
        self.kind = kind
        self.ref = ref
        self.parents = parents
        self.policy = policy
        self.revision = revision
        self.available = available
        self.last_sequence = last_sequence
        self.last_event = last_event


class ScopedSourceLifecycleService:
    """Register evidence dependencies and propagate Source availability events."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = _require_node_id(scope_id)

    async def register_source(self, node_id: str, ref: SourceRef, /, *, revision: int) -> None:
        """Register one Source node with its fixed lifecycle revision."""

        _require_positive("revision", revision)
        await self._register(
            node_id,
            KIND_SOURCE,
            ref.model_dump(mode="json"),
            parents=(),
            policy="all",
            revision=revision,
        )
        await self._reconcile(reason="source registered")

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        """Bind one Memory entry version to parent evidence nodes."""

        _require_policy(policy)
        parents = _require_parents(parents)
        await self._require_readable_memory(citation)
        await self._register(
            node_id,
            KIND_MEMORY,
            citation.model_dump(mode="json"),
            parents=parents,
            policy=policy,
            revision=1,
        )
        await self._reconcile(reason="memory registered")

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        """Bind one Topic Memory revision to parent evidence nodes."""

        _require_policy(policy)
        parents = _require_parents(parents)
        await self._require_readable_topic(reference)
        await self._register(
            node_id,
            KIND_TOPIC,
            reference.model_dump(mode="json"),
            parents=parents,
            policy=policy,
            revision=1,
        )
        await self._reconcile(reason="topic registered")

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source availability event and propagate to every descendant."""

        database = self._runtime._provider.database
        table = SOURCE_LIFECYCLE_TABLE
        async with database.transaction() as connection:
            node = await _load_row(connection, self.scope_id, event.node_id)
            if node is None:
                raise SourceLifecycleError(f"unknown lifecycle node {event.node_id!r}")  # noqa: TRY003
            if node.kind != KIND_SOURCE:
                message = f"lifecycle events only apply to Source nodes, not {event.node_id!r}"
                raise SourceLifecycleError(message)
            if event.revision != node.revision:
                message = f"event revision {event.revision} does not match lifecycle revision {node.revision}"
                raise SourceLifecycleError(message)
            if event.sequence == node.last_sequence:
                if node.last_event is not None and _same_event(node.last_event, event):
                    replay = True
                else:
                    message = f"sequence {event.sequence} was already used by a different event"
                    raise SourceLifecycleError(message)
            elif event.sequence > node.last_sequence:
                replay = False
            else:
                message = f"sequence {event.sequence} is not above the applied sequence {node.last_sequence}"
                raise SourceLifecycleError(message)
            if not replay:
                await connection.execute(
                    update(table)
                    .where(table.c.scope_id == self.scope_id, table.c.node_id == event.node_id)
                    .values(
                        available=event.available,
                        last_sequence=event.sequence,
                        last_event_payload=_dump_payload(event.model_dump(mode="json")),
                    )
                )
        await self._reconcile(reason=event.reason)
        nodes = await self._load_nodes()
        return sorted(node.node_id for node in nodes if node.available)

    async def resolve(self, node_id: str, /, *, purpose: str) -> Any:
        """Return the bound reference for generation or audit use."""

        if purpose not in _PURPOSES:
            message = f"unknown lifecycle purpose {purpose!r}"
            raise SourceLifecycleError(message)
        database = self._runtime._provider.database
        async with database.transaction() as connection:
            node = await _load_row(connection, self.scope_id, node_id)
        if node is None:
            message = f"unknown lifecycle node {node_id!r}"
            raise SourceLifecycleError(message)
        if purpose == "generation" and not node.available:
            message = f"lifecycle node {node_id!r} is withdrawn from generation use"
            raise SourceLifecycleError(message)
        if node.kind == KIND_SOURCE:
            return SourceRef.model_validate(node.ref)
        if node.kind == KIND_MEMORY:
            return MemoryCitation.model_validate(node.ref)
        return ArtifactRef.model_validate(node.ref)

    async def _register(
        self,
        node_id: str,
        kind: str,
        ref: dict[str, Any],
        *,
        parents: tuple[str, ...],
        policy: str,
        revision: int,
    ) -> None:
        database = self._runtime._provider.database
        table = SOURCE_LIFECYCLE_TABLE
        async with database.transaction() as connection:
            existing = await _load_row(connection, self.scope_id, node_id)
            if existing is not None:
                same = (
                    existing.kind == kind
                    and existing.ref == ref
                    and existing.parents == list(parents)
                    and existing.policy == policy
                    and existing.revision == revision
                )
                if not same:
                    raise SourceLifecycleError(f"lifecycle node {node_id!r} is already bound to a different contract")  # noqa: TRY003
                return
            for parent in parents:
                parent_row = await _load_row(connection, self.scope_id, parent)
                if parent_row is None:
                    raise SourceLifecycleError(f"parent lifecycle node {parent!r} is not registered")  # noqa: TRY003
            created_index = int(
                await connection.scalar(
                    select(func.max(table.c.created_index)).where(table.c.scope_id == self.scope_id)
                )
                or 0
            )
            await connection.execute(
                insert(table).values(
                    scope_id=self.scope_id,
                    node_id=node_id,
                    kind=kind,
                    ref_payload=_dump_payload(ref),
                    parents_payload=_dump_payload(list(parents)),
                    policy=policy,
                    revision=revision,
                    available=True,
                    last_sequence=0,
                    last_event_payload=None,
                    created_index=created_index + 1,
                )
            )

    async def _reconcile(self, *, reason: str) -> None:
        nodes = await self._load_nodes()
        availability = _propagate(nodes)
        changed = [node for node in nodes if node.kind != KIND_SOURCE and node.available != availability[node.node_id]]
        if changed:
            database = self._runtime._provider.database
            table = SOURCE_LIFECYCLE_TABLE
            async with database.transaction() as connection:
                for node in changed:
                    await connection.execute(
                        update(table)
                        .where(table.c.scope_id == self.scope_id, table.c.node_id == node.node_id)
                        .values(available=availability[node.node_id])
                    )
        memory_nodes = [node for node in nodes if node.kind == KIND_MEMORY]
        if not memory_nodes:
            return
        targets = {node.node_id: availability[node.node_id] for node in memory_nodes}
        await self._apply_memory_states(memory_nodes, targets, reason=reason)

    async def _apply_memory_states(self, nodes: list[_Node], targets: dict[str, bool], *, reason: str) -> None:
        runtime = self._runtime
        async with runtime._context(self.scope_id) as context, runtime._locked(self.scope_id):
            service = context.artifacts.memory
            try:
                current = await service.head(context.artifacts.memory_artifact_id)
            except ArtifactNotFoundError:
                return
            entries = await service.entries(current)
            versions = {entry.entry_id: entry for entry in entries}
            manifest_states = {
                item.entry_id: item.state for item in current.content.manifest.entries
            }
            to_deactivate: list[Any] = []
            to_reactivate: list[Any] = []
            for node in nodes:
                citation = MemoryCitation.model_validate(node.ref)
                entry = versions.get(citation.entry_id)
                if entry is None or entry.entry_version_id != citation.entry_version_id:
                    message = f"bound Memory entry {node.node_id!r} no longer matches its registered version"
                    raise SourceLifecycleError(message)
                state = manifest_states.get(citation.entry_id)
                if state == "active" and not targets[node.node_id]:
                    to_deactivate.append(entry)
                elif state != "active" and targets[node.node_id]:
                    to_reactivate.append(entry)
            if to_deactivate:
                current = await service.forget(current, entries=tuple(to_deactivate), reason=reason)
            if to_reactivate:
                await service.reactivate(current, entries=tuple(to_reactivate), reason=reason)

    async def _load_nodes(self) -> list[_Node]:
        database = self._runtime._provider.database
        async with database.transaction() as connection:
            table = SOURCE_LIFECYCLE_TABLE
            rows = (
                await connection.execute(
                    select(table).where(table.c.scope_id == self.scope_id).order_by(table.c.created_index)
                )
            ).mappings()
            return tuple(_node_from_row(row) for row in rows)

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        runtime = self._runtime
        async with runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            try:
                current = await service.head(context.artifacts.memory_artifact_id)
            except ArtifactNotFoundError as error:
                message = "cited Memory artifact does not exist"
                raise SourceLifecycleError(message) from error
            if citation.memory_ref.artifact_id != current.artifact_id:
                raise SourceLifecycleError("cited Memory artifact does not match the current scope artifact")  # noqa: TRY003
            for item in current.content.manifest.entries:
                if item.entry_id == citation.entry_id and item.entry_version_id == citation.entry_version_id:
                    return
            raise SourceLifecycleError("cited Memory entry version is not readable")  # noqa: TRY003

    async def _require_readable_topic(self, reference: ArtifactRef) -> None:
        records = self._runtime.records.for_scope(self.scope_id)
        try:
            await records.get_artifact_revision("topic-memory", reference.artifact_id, reference.revision)
        except Exception as error:
            message = "referenced Topic Memory revision is not readable"
            raise SourceLifecycleError(message) from error


async def withdrawn_topic_artifact_ids(database: Any, scope_id: str, /) -> frozenset[str]:
    """Return Topic artifact ids currently withdrawn from search and context."""

    table = SOURCE_LIFECYCLE_TABLE
    async with database.transaction() as connection:
        rows = await connection.execute(
            select(table.c.ref_payload).where(
                table.c.scope_id == scope_id,
                table.c.kind == KIND_TOPIC,
                table.c.available.is_(False),
            )
        )
        artifact_ids: set[str] = set()
        for (payload,) in rows:
            try:
                artifact_ids.add(json.loads(_load_payload_text(payload))["artifact_id"])
            except (ValueError, KeyError, TypeError):
                continue
        return frozenset(artifact_ids)


def _propagate(nodes: tuple[_Node, ...]) -> dict[str, bool]:
    availability: dict[str, bool] = {}
    for node in nodes:
        if node.kind == KIND_SOURCE:
            availability[node.node_id] = node.available
        else:
            support = [availability.get(parent, False) for parent in node.parents]
            availability[node.node_id] = all(support) if node.policy == "all" else any(support)
    return availability


def _node_from_row(row: Any) -> _Node:
    return _Node(
        node_id=row["node_id"],
        kind=row["kind"],
        ref=json.loads(_load_payload_text(row["ref_payload"])),
        parents=json.loads(_load_payload_text(row["parents_payload"])),
        policy=row["policy"],
        revision=int(row["revision"]),
        available=bool(row["available"]),
        last_sequence=int(row["last_sequence"]),
        last_event=None
        if row["last_event_payload"] is None
        else json.loads(_load_payload_text(row["last_event_payload"])),
    )


async def _load_row(connection: Any, scope_id: str, node_id: str) -> Any:
    table = SOURCE_LIFECYCLE_TABLE
    condition = (table.c.scope_id == scope_id) & (table.c.node_id == node_id)
    if connection.dialect.name == "sqlite":
        await connection.execute(update(table).where(condition).values(available=table.c.available))
    row = (await connection.execute(select(table).where(condition).with_for_update())).mappings().one_or_none()
    return None if row is None else _node_from_row(row)


def _same_event(payload: dict[str, Any], event: SourceLifecycleEvent) -> bool:
    return (
        payload.get("node_id") == event.node_id
        and payload.get("revision") == event.revision
        and payload.get("sequence") == event.sequence
        and payload.get("available") == event.available
        and payload.get("reason") == event.reason
    )


def _dump_payload(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def _load_payload_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


def _require_node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        message = "node_id must be a non-empty trimmed identifier"
        raise SourceLifecycleError(message)
    if len(value) > MAX_SOURCE_ID_LENGTH:
        message = "node_id exceeds the identifier length limit"
        raise SourceLifecycleError(message)
    return value


def _require_policy(policy: str) -> None:
    if policy not in ("all", "any"):
        message = "policy must be 'all' or 'any'"
        raise SourceLifecycleError(message)


def _require_parents(parents: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(parents, tuple) or not parents:
        message = "derived nodes require at least one parent"
        raise SourceLifecycleError(message)
    seen: set[str] = set()
    for parent in parents:
        _require_node_id(parent)
        if parent in seen:
            message = "parents must be unique"
            raise SourceLifecycleError(message)
        seen.add(parent)
    return parents


def _require_positive(field: str, value: int) -> None:
    if type(value) is not int or value < 1:
        message = f"{field} must be a positive integer"
        raise SourceLifecycleError(message)


__all__ = [
    "KIND_MEMORY",
    "KIND_SOURCE",
    "KIND_TOPIC",
    "ScopedSourceLifecycleService",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "withdrawn_topic_artifact_ids",
]
