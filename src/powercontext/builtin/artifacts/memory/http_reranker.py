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

"""HTTP score-service adapter for Memory reranking with global rank ordering."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANKER_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_HTTP_RERANKER_BATCH_SIZE = 32


class HTTPMemoryReranker:
    """Rerank Memory candidates through one offline HTTP scoring service.

    The candidate pool keeps its global one-based ranks across batches, every rank is
    scored exactly once, and the final selection combines all batches by score with a
    stable original-rank tiebreak.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANKER_BATCH_SIZE,
        policy_id: str = DEFAULT_HTTP_RERANKER_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint:
            message = "endpoint must be a non-empty URL string"
            raise ValueError(message)
        if type(batch_size) is not int or batch_size <= 0:
            message = "batch_size must be a positive integer"
            raise ValueError(message)
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select the globally best ranks."""

        candidate_count = len(candidates)
        if candidate_count < 1:
            message = "rerank requires at least one candidate"
            raise ValueError(message)
        if type(limit) is not int or limit < 1 or limit > candidate_count:
            message = "rerank limit must be between one and the candidate count"
            raise ValueError(message)

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, candidate_count, self._batch_size):
            batch = tuple(candidates[start : start + self._batch_size])
            batch_scores = await self._score_batch(query, batch, start)
            requests += 1
            scores.update(batch_scores)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _score in ranked[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))

    async def _score_batch(
        self,
        query: str,
        batch: tuple[MemoryHit, ...],
        start: int,
        /,
    ) -> dict[int, float]:
        payload = {
            "query": query,
            "candidates": [
                {"rank": start + offset + 1, "text": candidate.text}
                for offset, candidate in enumerate(batch)
            ],
        }
        response = await self._client.post(self._endpoint, json=payload)
        response.raise_for_status()
        try:
            document = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            message = "reranker response must be UTF-8 JSON"
            raise ValueError(message) from error
        return _parse_scores(document, expected_ranks=tuple(start + offset + 1 for offset in range(len(batch))))


def _parse_scores(document: Any, *, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    if not isinstance(document, dict) or set(document) != {"scores"}:
        message = "reranker response must contain exactly a scores field"
        raise ValueError(message)
    entries = document["scores"]
    if not isinstance(entries, list) or len(entries) != len(expected_ranks):
        message = "reranker scores must cover every batch rank exactly once"
        raise ValueError(message)
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            message = "each score entry must contain exactly rank and score"
            raise ValueError(message)
        rank = entry["rank"]
        score = entry["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in scores:
            message = "score ranks must match the batch ranks exactly once"
            raise ValueError(message)
        if type(score) not in (int, float) or not math.isfinite(score):
            message = "scores must be finite numbers"
            raise ValueError(message)
        scores[rank] = float(score)
    return scores


__all__ = [
    "DEFAULT_HTTP_RERANKER_BATCH_SIZE",
    "DEFAULT_HTTP_RERANKER_POLICY_ID",
    "HTTPMemoryReranker",
]
