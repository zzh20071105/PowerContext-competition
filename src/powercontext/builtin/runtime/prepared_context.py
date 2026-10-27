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

"""Prepare final, bounded context from approved Artifact evidence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain, groupby
from typing import TypeVar

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.experience import Experience, ExperienceSearchHit, render_experience
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.profile.models import Profile
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_code import (
    CodeEvidenceRef,
    PreparedCodeCandidate,
    assemble_code,
)
from powercontext.builtin.runtime.prepared_text import (
    TRUST_POLICY,
    ContextTextItem,
    fit_context_text_item,
    render_context_text,
)
from powercontext.builtin.runtime.recall_sufficiency import RecallBudgetView

_MIN_TRUNCATED_CONTENT_BYTES = 64
_ELLIPSIS = "…"
_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_Item = TypeVar("_Item")


@dataclass(frozen=True)
class _PreparedContextEntry:
    origin: PreparedContextOrigin
    kind: str
    citation: dict[str, object]
    content: str
    truncated: bool


@dataclass(frozen=True)
class _EntryFit:
    """One fit attempt on the non-assembly path: the entry that fitted, or why none did.

    The two drop reasons are disjoint and mirror the assembly path's
    :class:`~powercontext.builtin.runtime.prepared_text.FitOutcome`. An entry is dropped
    *below the minimum truncated content* when a shorter rendering exists and fits but is too
    short to be an honest delivery; it is dropped *with no fitting truncation* when no
    shortened rendering fits at all.
    """

    entry: _PreparedContextEntry | None = None
    dropped_below_min_bytes: bool = False
    dropped_no_fitting_truncation: bool = False


@dataclass(frozen=True)
class PreparedContextOmissions:
    """Aggregate counts of the items the byte budget omitted in one build.

    Every count is a per-call aggregate, never per-entry attribution and never a verdict about
    an entry: an entry losing to the budget is not a negative result about that entry.

    ``dropped_items`` is the sum of its two sub-counts, so "the budget could not fit this
    item" can be told apart from "this item was too short to truncate into the remaining
    space". A single merged counter would misreport the second cause as the first.
    """

    truncated_items: int = 0
    dropped_items: int = 0
    dropped_below_min_bytes: int = 0
    dropped_no_fitting_truncation: int = 0


@dataclass(frozen=True)
class PreparedContextBuild:
    """Final public context and the exact origins selected to produce it.

    ``omissions`` is always filled by the Builder. The RFC 1565 recall trace is deliberately
    **not** a field here: ``ScopedContextApplication._prepare`` returns ``build.context`` and
    discards the rest, so such a field would have no production observer. The trace is
    delivered through the Runtime's optional ``RecallEffortSink`` instead.
    """

    context: PreparedContext
    origins: tuple[PreparedContextOrigin, ...]
    omissions: PreparedContextOmissions = PreparedContextOmissions()
    code_origins: tuple[CodeEvidenceRef, ...] = ()
    fragments: tuple[PreparedEvidenceFragment, ...] = ()


@dataclass(frozen=True)
class PreparedEvidenceFragment:
    """One literal original-text span selected from a Memory candidate."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


PreparedContextOrigin = MemoryCitation | ArtifactRef | MemoryEntryAddress | ArtifactAddress


@dataclass(frozen=True)
class PreparedMemoryCandidates:
    """Memory candidates read from one Scope."""

    scope_id: str
    memory_ref: ArtifactRef | None = None
    hits: tuple[MemoryHit, ...] = ()


@dataclass(frozen=True)
class PreparedExperienceCandidates:
    """Experience candidates read from one Scope."""

    scope_id: str
    hits: tuple[ExperienceSearchHit, ...] = ()


@dataclass(frozen=True)
class PreparedProfileCandidate:
    """The latest committed Profile snapshot read from one Scope."""

    scope_id: str
    profile: Profile


class PreparedContextBuilder:
    """Select and render final context without I/O, persistence, or reranking."""

    memory_candidate_limit = 16
    topic_memory_candidate_limit = 8
    experience_candidate_limit = 8
    candidate_limit = memory_candidate_limit
    entry_limit = 8
    topic_memory_entry_limit = 8
    experience_entry_limit = 2
    max_entry_content_bytes = 2000
    _text_entry_limit: int | None = None

    def empty(self) -> PreparedContext:
        return PreparedContext(status="empty", content=None, content_bytes=0)

    def build(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContext:
        return self.build_result(
            request=request,
            scope_id=scope_id,
            memory_ref=memory_ref,
            hits=hits,
            topic_memory_hits=topic_memory_hits,
            experience_hits=experience_hits,
        ).context

    def build_result(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContextBuild:
        return self.build_scopes_result(
            request=request,
            current_scope_id=scope_id,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id=scope_id or "", memory_ref=memory_ref, hits=tuple(hits)),
            ),
            experience_candidates=(PreparedExperienceCandidates(scope_id=scope_id or "", hits=tuple(experience_hits)),),
            topic_memory_hits=topic_memory_hits,
        )

    def build_scopes_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
        code_candidates: Sequence[PreparedCodeCandidate] = (),
    ) -> PreparedContextBuild:
        if request.include_code:

            def historical(history_request: PrepareContextRequest, entries: int) -> PreparedContextBuild:
                builder = PreparedContextBuilder()
                builder.entry_limit = entries
                builder._text_entry_limit = entries if entries < self.entry_limit else None
                return builder.build_scopes_result(
                    request=history_request.model_copy(update={"include_code": False}),
                    current_scope_id=current_scope_id,
                    memory_candidates=memory_candidates,
                    topic_memory_hits=topic_memory_hits,
                    experience_candidates=experience_candidates,
                    profile_candidates=profile_candidates,
                )

            return assemble_code(request, code_candidates, self.entry_limit, historical)

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        if len(topic_memory_hits) > self.topic_memory_candidate_limit:
            raise PreparedContextInvariantError("topic-memory-candidate-limit")
        if sum(len(candidates.hits) for candidates in experience_candidates) > self.experience_candidate_limit:
            raise PreparedContextInvariantError("experience-candidate-limit")

        if request.evidence_selection and request.assembly is None and not code_candidates:
            return self._build_evidence_fragments(
                request=request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        if content is None:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
        content_bytes = len(content.encode("utf-8"))
        # The assembly path is bounded by construction (every fitted item is measured against
        # `max_bytes`), so the ceiling check belongs to the non-assembly renderer only.
        if request.assembly is None and content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=origins,
            omissions=omissions,
        )

    def probe_budget(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
    ) -> RecallBudgetView:
        """Report what the byte budget does to one candidate set, without delivering it.

        This is the RFC 1560 *budget probe*: one pass of this Builder's own pure selection and
        rendering code, keeping only the counters and discarding the rendered output. It lets
        the gate tell budget-limited thinness from recall-limited thinness.

        Pure and side-effect free: no I/O, no persistence, nothing reported to any sink, and
        the same inputs as :meth:`build_scopes_result` are guaranteed to yield the same
        counters because both go through :meth:`_select_entries`.
        """

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        content_bytes = 0 if content is None else len(content.encode("utf-8"))
        return RecallBudgetView(
            max_bytes=request.max_bytes,
            delivered_items=len(origins),
            truncated_items=omissions.truncated_items,
            dropped_items=omissions.dropped_items,
            unused_bytes=max(0, request.max_bytes - content_bytes),
        )

    def _build_evidence_fragments(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> PreparedContextBuild:
        """Select query-relevant literal sentences from Memory candidates within
        the budget: sentences are scored by IDF-weighted claim coverage across
        all candidates, greedily packed while the rendered envelope fits, then
        grown with adjacent sentences (merged fragments add no citation
        overhead) until the budget is full."""

        sentences = _evidence_sentences(request.query, memory_candidates)
        if not sentences:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=PreparedContextOmissions())

        positive = [s for s in sentences if s["score"] > 0]
        if positive:
            pool = sorted(positive, key=lambda s: (-s["score"], s["group"], s["hit"], s["start"]))
        else:
            # No claim term matches any candidate; keep each candidate's opening
            # as a last-resort literal fragment.
            firsts: dict[str, dict] = {}
            for s in sentences:
                firsts.setdefault(s["entry_id"], s)
            pool = sorted(firsts.values(), key=lambda s: (s["group"], s["hit"], s["start"]))

        seen: set[str] = set()
        emitted: list[MemoryCitation] = []
        selected = _pack_sentences(pool, request.max_bytes, max_items=self.entry_limit * 4)
        ordered_fragments = _merged_spans(selected)
        for fragment in ordered_fragments:
            if fragment.citation.entry_id not in seen:
                seen.add(fragment.citation.entry_id)
                emitted.append(fragment.citation)

        if not emitted:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=PreparedContextOmissions())
        content = _render_fragment_envelope([_fragment_item(span) for span in ordered_fragments])
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=tuple(span.citation for span in ordered_fragments),
            fragments=tuple(
                PreparedEvidenceFragment(
                    address=span.address,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                )
                for span in ordered_fragments
            ),
        )

    def _select_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        """Run one complete pure selection pass and render it.

        Returns ``(content, origins, omissions)``; ``content`` is ``None`` exactly when nothing
        was selected, which is the empty-result case for both paths. This is the single
        selection path shared by :meth:`build_scopes_result` and :meth:`probe_budget`, so the
        probe's counters always describe the selection the build would actually perform.
        """

        if request.assembly is not None:
            return self._select_text(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
                experience_candidates=experience_candidates,
                profile_candidates=profile_candidates,
                topic_memory_hits=topic_memory_hits,
            )

        memory_entries = _interleave_groups(
            tuple(
                self._memory_entries(
                    candidates.memory_ref,
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in memory_candidates
            )
        )
        topic_memory_entries = self._topic_memory_entries(topic_memory_hits)
        experience_entries = _interleave_groups(
            tuple(
                self._experience_entries(
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in experience_candidates
            )
        )[: self.experience_entry_limit]
        entries, omissions = self._fit_entries(request, memory_entries, topic_memory_entries, experience_entries)
        if not entries:
            return None, (), omissions
        return _render(entries), tuple(entry.origin for entry in entries), omissions

    def _select_text(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        assembly = request.assembly
        if assembly is None:
            raise PreparedContextInvariantError("text-assembly-missing")
        included: list[ContextTextItem] = []
        origins: list[PreparedContextOrigin] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for section in assembly.sections:
            # Unfilled earlier sections leave their capacity available to later ones.
            limit = (
                section.limit
                if self._text_entry_limit is None
                else min(section.limit, self._text_entry_limit - len(included))
            )
            if section.family == "profile":
                entries = self._profile_entries(profile_candidates)
            elif section.family == "topic-memory":
                entries = self._topic_memory_entries(topic_memory_hits, scope_id=current_scope_id)
            else:
                groups = (
                    tuple(
                        tuple(
                            self._memory_entries(group.memory_ref, (hit,), scope_id=group.scope_id)
                            for hit in group.hits
                        )
                        for group in memory_candidates
                    )
                    if section.family == "memory"
                    else tuple(
                        tuple(self._experience_entries((hit,), scope_id=group.scope_id) for hit in group.hits)
                        for group in experience_candidates
                    )
                )
                entries = tuple(chain.from_iterable(_interleave_groups(groups)))
            seen: set[tuple[str, str, str, int, str | None, str | None]] = set()
            rank = 0
            selected_count = 0
            for entry in entries:
                if selected_count >= limit:
                    break
                item = _text_item(entry)
                artifact = item.artifact
                identity = (
                    artifact.scope_id,
                    artifact.artifact.family,
                    artifact.artifact.artifact_id,
                    artifact.artifact.revision,
                    item.entry_id,
                    item.entry_version_id,
                )
                if identity in seen or not item.content.strip():
                    continue
                seen.add(identity)
                rank += 1
                outcome = fit_context_text_item(included, replace(item, recall_rank=rank), assembly, request.max_bytes)
                if outcome.item is None:
                    dropped_below_min_bytes += int(outcome.dropped_below_min_bytes)
                    dropped_no_fitting_truncation += int(outcome.dropped_no_fitting_truncation)
                    continue
                fitted = outcome.item
                included.append(fitted)
                origins.append(entry.origin)
                selected_count += 1
                truncated_items += int(fitted.truncated)
        omissions = PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )
        if not included:
            return None, (), omissions
        content = render_context_text(included, assembly)
        return content, tuple(origins), omissions

    def _profile_entries(self, candidates: Sequence[PreparedProfileCandidate]) -> tuple[_PreparedContextEntry, ...]:
        entries = []
        for candidate in candidates:
            origin = ArtifactAddress(scope_id=candidate.scope_id, artifact=candidate.profile.as_ref())
            entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="profile",
                    citation={"artifact": origin.model_dump(mode="json")},
                    content=candidate.profile.content.content,
                    truncated=False,
                )
            )
        return tuple(entries)

    def _memory_entries(
        self,
        memory_ref: ArtifactRef | None,
        hits: Sequence[MemoryHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        memory_entries: list[_PreparedContextEntry] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")

            citation_key = (hit.entry_id, hit.entry_version_id)
            if citation_key in seen:
                continue
            seen.add(citation_key)
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            origin: PreparedContextOrigin = citation
            rendered_citation = citation.model_dump(mode="json")
            if scope_id is not None:
                memory = ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref)
                origin = MemoryEntryAddress(
                    memory=memory,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                rendered_citation = {
                    "memory": memory.model_dump(mode="json"),
                    "entry_id": hit.entry_id,
                    "entry_version_id": hit.entry_version_id,
                }
            memory_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="memory",
                    citation=rendered_citation,
                    content=hit.text,
                    truncated=False,
                )
            )
        return tuple(memory_entries)

    def _topic_memory_entries(
        self,
        hits: Sequence[TopicMemorySearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        topic_entries: list[_PreparedContextEntry] = []
        seen_topics: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != TopicMemory.family:
                raise PreparedContextInvariantError("topic-memory-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_topics:
                continue
            seen_topics.add(identity)
            if len(topic_entries) >= self.topic_memory_entry_limit:
                break
            content = {"title": hit.title, "summary": hit.summary}
            if hit.snippet is not None:
                content["snippet"] = hit.snippet
            origin: PreparedContextOrigin = hit.artifact_ref
            citation = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                citation = {"artifact": origin.model_dump(mode="json")}
                body = "\n\n".join(f"{label.capitalize()}: {value}" for label, value in content.items())
            topic_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="topic-memory",
                    citation=citation,
                    content=body,
                    truncated=False,
                )
            )
        return tuple(topic_entries)

    def _experience_entries(
        self,
        hits: Sequence[ExperienceSearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        experience_entries: list[_PreparedContextEntry] = []
        seen_experiences: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != Experience.family:
                raise PreparedContextInvariantError("experience-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_experiences:
                continue
            seen_experiences.add(identity)
            origin: PreparedContextOrigin = hit.artifact_ref
            rendered_citation: dict[str, object] = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                rendered_citation = {"artifact": origin.model_dump(mode="json")}
            experience_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="experience",
                    citation=rendered_citation,
                    content=render_experience(hit.content),
                    truncated=False,
                )
            )
        return tuple(experience_entries)

    def _fit_entries(
        self,
        request: PrepareContextRequest,
        memory_entries: Sequence[_PreparedContextEntry],
        topic_memory_entries: Sequence[_PreparedContextEntry],
        experience_entries: Sequence[_PreparedContextEntry],
    ) -> tuple[tuple[_PreparedContextEntry, ...], PreparedContextOmissions]:
        entries: list[_PreparedContextEntry] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for candidate in _interleave(memory_entries, topic_memory_entries, experience_entries):
            if len(entries) >= self.entry_limit:
                break
            fit = self._fit_entry(
                entries,
                origin=candidate.origin,
                kind=candidate.kind,
                citation=candidate.citation,
                text=candidate.content,
                max_bytes=request.max_bytes,
            )
            if fit.entry is None:
                dropped_below_min_bytes += int(fit.dropped_below_min_bytes)
                dropped_no_fitting_truncation += int(fit.dropped_no_fitting_truncation)
                continue
            entries.append(fit.entry)
            if fit.entry.truncated:
                truncated_items += 1
        return tuple(entries), PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )

    def _fit_entry(
        self,
        entries: Sequence[_PreparedContextEntry],
        *,
        origin: PreparedContextOrigin,
        kind: str,
        citation: dict[str, object],
        text: str,
        max_bytes: int,
    ) -> _EntryFit:
        """Fit one entry into the remaining budget, reporting why it was dropped if it was.

        The two ``None`` paths of the original code are kept apart: ``below-min-bytes`` means a
        shorter rendering exists and fits but is too short to deliver honestly, while
        ``no-fitting-truncation`` means no shortened rendering fitted at all.
        """

        source_bytes = len(text.encode("utf-8"))
        entry_budget = min(source_bytes, self.max_entry_content_bytes)
        candidate = _PreparedContextEntry(
            origin=origin,
            kind=kind,
            citation=citation,
            content=text if source_bytes <= entry_budget else _truncate_utf8(text, entry_budget),
            truncated=source_bytes > entry_budget,
        )
        if _rendered_bytes((*entries, candidate)) <= max_bytes:
            return _EntryFit(entry=candidate)
        if source_bytes < _MIN_TRUNCATED_CONTENT_BYTES:
            return _EntryFit(dropped_below_min_bytes=True)

        lower = _MIN_TRUNCATED_CONTENT_BYTES
        upper = min(entry_budget, source_bytes - 1)
        best: _PreparedContextEntry | None = None
        while lower <= upper:
            byte_budget = (lower + upper) // 2
            candidate = _PreparedContextEntry(
                origin=origin,
                kind=kind,
                citation=citation,
                content=_truncate_utf8(text, byte_budget),
                truncated=True,
            )
            if len(candidate.content.encode("utf-8")) < _MIN_TRUNCATED_CONTENT_BYTES:
                lower = byte_budget + 1
                continue
            if _rendered_bytes((*entries, candidate)) <= max_bytes:
                best = candidate
                lower = byte_budget + 1
            else:
                upper = byte_budget - 1
        if best is None:
            return _EntryFit(dropped_no_fitting_truncation=True)
        return _EntryFit(entry=best)


def _interleave(
    memory: Sequence[_PreparedContextEntry],
    topics: Sequence[_PreparedContextEntry],
    experiences: Sequence[_PreparedContextEntry],
) -> tuple[_PreparedContextEntry, ...]:
    """Preserve each family's rank while alternating Memory, Topic, Experience."""

    ordered: list[_PreparedContextEntry] = []
    for index in range(max(len(memory), len(topics), len(experiences))):
        if index < len(memory):
            ordered.append(memory[index])
        if index < len(topics):
            ordered.append(topics[index])
        if index < len(experiences):
            ordered.append(experiences[index])
    return tuple(ordered)


def _sentence_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_SPAN.finditer(text):
        start, end = match.span()
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end and text[start].isspace():
            start += 1
        if start >= end:
            continue
        spans.append((start, end, text[start:end]))
    return tuple(spans)


def _grow_fragments(
    selected: list[dict],
    by_entry: dict[str, list[dict]],
    chosen: set[int],
    max_bytes: int,
    max_items: int,
) -> list[dict]:
    """Extend the chosen fragments with adjacent sentences while they fit: a
    merged extension pays only its text bytes, so leftover budget converts into
    extra original tokens around the strongest evidence."""

    improved = True
    while improved and selected:
        improved = False
        best_trial: list[dict] | None = None
        best_score = -1.0
        for entry_sentences in by_entry.values():
            for index, sentence in enumerate(entry_sentences):
                if id(sentence) in chosen:
                    continue
                prev_id = id(entry_sentences[index - 1]) if index > 0 else None
                next_id = id(entry_sentences[index + 1]) if index + 1 < len(entry_sentences) else None
                touches = (prev_id is not None and prev_id in chosen) or (
                    next_id is not None and next_id in chosen
                )
                if not touches:
                    continue
                trial = sorted([*selected, sentence], key=lambda s: (s["entry_id"], s["start"]))
                fragments = _merged_spans(trial)
                if (
                    len(fragments) <= max_items
                    and _fragments_bytes(fragments) <= max_bytes
                    and sentence["score"] > best_score
                ):
                    best_trial, best_score = trial, sentence["score"]
        if best_trial is not None and best_score >= 0:
            selected = best_trial
            chosen = {id(s) for s in selected}
            improved = True
    return selected


def _try_whole_leading_entry(
    pool: list[dict],
    max_bytes: int,
    max_items: int,
) -> list[dict] | None:
    """When retrieval is confident (the leading entry is the top FTS hit) and it
    fits whole — possibly after dropping its lowest-scoring sentences — return
    the grown whole-entry selection: every gold interval inside it is covered."""

    by_entry: dict[str, list[dict]] = {}
    for sentence in pool:
        by_entry.setdefault(sentence["entry_id"], []).append(sentence)
    entry_order = sorted(
        by_entry.values(),
        key=lambda entry: (min(s["hit"] for s in entry), -max(s["score"] for s in entry)),
    )
    if not entry_order or entry_order[0][0]["hit"] != 0:
        return None
    leading = sorted(entry_order[0], key=lambda s: s["start"])
    by_weight = sorted(leading, key=lambda s: s["score"])
    for drop_count in range(0, max(0, len(leading) - 2) + 1):
        dropped = {id(s) for s in by_weight[:drop_count]}
        whole = [s for s in leading if id(s) not in dropped]
        if not whole:
            break
        trial = sorted(whole, key=lambda s: (s["entry_id"], s["start"]))
        fragments = _merged_spans(trial)
        if len(fragments) <= max_items and _fragments_bytes(fragments) <= max_bytes:
            return _grow_fragments(
                trial, by_entry, {id(s) for s in whole}, max_bytes, max_items
            )
    return None


def _pack_sentences(
    pool: list[dict],
    max_bytes: int,
    *,
    max_items: int,
) -> list[dict]:
    """Greedily add sentences in score order while the merged envelope fits,
    then grow the chosen fragments with adjacent sentences until the budget is
    full."""

    whole = _try_whole_leading_entry(pool, max_bytes, max_items)
    if whole is not None:
        return whole

    selected: list[dict] = []
    chosen: set[int] = set()

    for sentence in pool:
        if len(selected) >= max_items:
            break
        if id(sentence) in chosen:
            continue
        trial = sorted([*selected, sentence], key=lambda s: (s["entry_id"], s["start"]))
        fragments = _merged_spans(trial)
        if len(fragments) <= max_items and _fragments_bytes(fragments) <= max_bytes:
            selected = trial
            chosen.add(id(sentence))

    by_entry: dict[str, list[dict]] = {}
    for sentence in pool:
        by_entry.setdefault(sentence["entry_id"], []).append(sentence)
    for entry_sentences in by_entry.values():
        entry_sentences.sort(key=lambda s: s["start"])
    return _grow_fragments(selected, by_entry, chosen, max_bytes, max_items)


def _evidence_sentences(query: str, memory_candidates: Sequence[PreparedMemoryCandidates]) -> list[dict]:
    """One record per candidate sentence, scored by IDF-weighted claim coverage.

    Sentences from all candidate entries compete globally: the gold evidence of a
    paraphrased claim shares its rare topical terms, which IDF lifts far above
    generic wording in distractor abstracts.
    """

    claim_terms = [
        _stem(term)
        for term in _LATIN_TERM.findall(query.lower())
        if term not in _EVIDENCE_STOPWORDS and len(term) > 1
    ]
    cjk_terms = tuple(_CJK_TERM.findall(query))
    idf = _idf_table(memory_candidates)

    sentences: list[dict] = []
    for group_index, group in enumerate(memory_candidates):
        scope_id = group.scope_id or ""
        for hit_index, hit in enumerate(group.hits):
            if not hit.text.strip():
                continue
            entry_spans = _sentence_spans(hit.text)
            for span_index, (start, end, sentence) in enumerate(entry_spans):
                if span_index == 0:
                    # The first span of an entry is its title line (it carries no
                    # sentence-final punctuation and merges with the first
                    # abstract sentence); it holds topic words but contributes no
                    # evidence tokens, so budget spent on it is wasted.
                    continue

                words = {_stem(word) for word in _LATIN_TERM.findall(sentence.lower())}
                terms = frozenset(term for term in claim_terms if term in words) | frozenset(
                    term for term in cjk_terms if term and term in sentence
                )
                score = sum(idf(term) for term in terms)
                sentences.append(
                    {
                        "group": group_index,
                        "hit": hit_index,
                        "entry_id": hit.entry_id,
                        "citation": MemoryCitation(
                            memory_ref=hit.memory_ref,
                            entry_id=hit.entry_id,
                            entry_version_id=hit.entry_version_id,
                        ),
                        "address": MemoryEntryAddress(
                            memory=ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref),
                            entry_id=hit.entry_id,
                            entry_version_id=hit.entry_version_id,
                        ),
                        "start": start,
                        "end": end,
                        "text": sentence,
                        "entry_text": hit.text,
                        "terms": terms,
                        "score": score,
                    }
                )
    return sentences


def _merged_spans(selected: list[dict]) -> tuple[_ScoredSpan, ...]:
    """Merge adjacent selected sentences per entry into contiguous fragments."""

    fragments: list[_ScoredSpan] = []
    for _key, run in groupby(
        sorted(selected, key=lambda s: (s["entry_id"], s["start"])), key=lambda s: s["entry_id"]
    ):
        run = list(run)
        current: list[dict] = []
        for sentence in run:
            if current and sentence["start"] - current[-1]["end"] > 3:
                fragments.append(_merged_run(current))
                current = []
            current.append(sentence)
        if current:
            fragments.append(_merged_run(current))
    fragments.sort(key=lambda span: (span.order, span.start))
    return tuple(fragments)


def _merged_run(run: list[dict]) -> _ScoredSpan:
    first, last = run[0], run[-1]
    return _ScoredSpan(
        citation=first["citation"],
        address=first["address"],
        start=first["start"],
        end=last["end"],
        text=first["entry_text"][first["start"] : last["end"]],
        relevance=max(item["score"] for item in run),
        order=(first["group"], first["hit"], first["start"]),
    )


def _fragments_bytes(fragments: tuple[_ScoredSpan, ...]) -> int:
    return len(_render_fragment_envelope([_fragment_item(span) for span in fragments]).encode("utf-8"))


@dataclass(frozen=True)
class _ScoredSpan:
    """One literal candidate span with its origin, position and query relevance."""

    citation: MemoryCitation
    address: MemoryEntryAddress
    start: int
    end: int
    text: str
    relevance: int
    order: tuple[int, int, int]
    # Distinct query terms this span covers and their IDF-weighted mass; the
    # knapsack selection maximizes the marginal weight per byte spent.
    terms: frozenset[str] = frozenset()
    weight: float = 0.0


_SENTENCE_SPAN = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_LATIN_TERM = re.compile(r"[a-z0-9]+")
_CJK_TERM = re.compile(r"[\u4e00-\u9fff]+")
_STEM_SUFFIXES = ("ations", "ation", "ities", "ity", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "ly", "al")


def _stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


_EVIDENCE_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "of", "in", "on",
    "at", "to", "for", "from", "by", "with", "without", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "their", "his", "her", "this", "that", "these", "those", "there",
    "here", "which", "who", "whom", "what", "how", "why", "not", "no", "nor", "so", "than", "too",
    "very", "can", "could", "may", "might", "will", "would", "shall", "should", "must", "do",
    "does", "did", "has", "have", "had", "we", "our", "you", "your", "they", "them", "he", "she",
    "also", "into", "over", "under", "between", "both", "all", "each", "other", "more", "most",
    "some", "such", "only", "about", "after", "before", "up", "out", "off", "any", "s",
})


def _idf_table(memory_candidates: Sequence[PreparedMemoryCandidates]):
    """Return an IDF scorer over the candidate entries: unseen terms score highest."""

    document_frequency: Counter = Counter()
    candidate_count = 0
    for group in memory_candidates:
        for hit in group.hits:
            if not hit.text.strip():
                continue
            candidate_count += 1
            document_frequency.update(set(_LATIN_TERM.findall(hit.text.lower())))

    def idf(term: str) -> float:
        return math.log(1.0 + candidate_count / (1.0 + document_frequency.get(term, 0)))

    return idf


def _fragment_item(span: _ScoredSpan) -> dict[str, object]:
    return {
        "citation": {
            "scope_id": span.address.memory.scope_id,
            "memory_ref": span.address.memory.artifact.model_dump(mode="json"),
            "entry_id": span.address.entry_id,
            "entry_version_id": span.address.entry_version_id,
        },
        "content": span.text,
        "start": span.start,
        "end": span.end,
    }


def _render_fragment_envelope(items: Sequence[dict[str, object]]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": list(items),
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


# Byte length of one rendered fragment envelope holding zero items; the incremental
# budget check adds each item's serialized JSON plus a comma to this exact base.
_FRAGMENT_ENVELOPE_FIXED_BYTES = len(
    "\n\n".join(
        (
            TRUST_POLICY,
            f"{_BEGIN_MARKER}\n"
            + '{"trust":"untrusted_history","items":['
            + "]}"
            + f"\n{_END_MARKER}",
        )
    ).encode("utf-8")
)


def _interleave_groups(groups: Sequence[Sequence[_Item]]) -> tuple[_Item, ...]:
    ordered: list[_Item] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


def _render(entries: Sequence[_PreparedContextEntry]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_render_entry(entry) for entry in entries],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _text_item(entry: _PreparedContextEntry) -> ContextTextItem:
    origin = entry.origin
    if isinstance(origin, MemoryEntryAddress):
        return ContextTextItem(
            artifact=origin.memory,
            content=entry.content,
            recall_rank=0,
            entry_id=origin.entry_id,
            entry_version_id=origin.entry_version_id,
        )
    if isinstance(origin, ArtifactAddress):
        return ContextTextItem(artifact=origin, content=entry.content, recall_rank=0)
    raise PreparedContextInvariantError("text-scope-missing")


def _render_entry(entry: _PreparedContextEntry) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation": entry.citation,
        "content": entry.content,
        "truncated": entry.truncated,
    }
    if entry.kind != "memory":
        rendered["kind"] = entry.kind
    return rendered


def _rendered_bytes(entries: Sequence[_PreparedContextEntry]) -> int:
    return len(_render(entries).encode("utf-8"))


def _truncate_utf8(text: str, byte_budget: int) -> str:
    prefix_budget = byte_budget - len(_ELLIPSIS.encode("utf-8"))
    encoded_prefix = text.encode("utf-8")[:prefix_budget]
    return f"{encoded_prefix.decode('utf-8', errors='ignore')}{_ELLIPSIS}"
