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

"""Cross-call Embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from powercontext.builtin.inference import EmbeddingResult, InferenceUsage

_DEFAULT_MAX_BATCH_SIZE = 64
_DEFAULT_COALESCE_MS = 5.0
_DEFAULT_CACHE_SIZE = 1024


class CoalescingEmbeddingModel:
    """Share in-flight Embedding work and successful vectors across calls.

    Identical texts inside one call, inside the coalescing window, or already awaited
    by an executing request share one underlying computation. Each call is restored to
    its original input order and reports the provider usage of exactly the distinct
    underlying batches it waited for; cache hits contribute no requests.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = _DEFAULT_COALESCE_MS,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            message = "max_batch_size must be a positive integer"
            raise ValueError(message)
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            message = "coalesce_ms must be a non-negative number"
            raise ValueError(message)
        if type(cache_size) is not int or cache_size < 0:
            message = "cache_size must be a non-negative integer"
            raise ValueError(message)
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._waiters: dict[str, asyncio.Future] = {}
        self._pending: list[str] = []
        self._batch_sequence = 0
        self._lock: asyncio.Lock | None = None
        self._work: asyncio.Event | None = None
        self._dispatcher: asyncio.Task | None = None

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        """Return vectors in input order, sharing and caching identical texts."""

        self._ensure_loop_resources()
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        unique_texts: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        async with self._lock:
            vectors: dict[str, Any] = {}
            futures: list[tuple[str, asyncio.Future]] = []
            for text in unique_texts:
                cached = self._cache.get(text) if self._cache_size else None
                if cached is not None:
                    self._cache.move_to_end(text)
                    vectors[text] = cached
                    continue
                future = self._waiters.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self._waiters[text] = future
                    self._pending.append(text)
                futures.append((text, future))
            if self._pending:
                self._work.set()

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        counted_batches: set[int] = set()
        for text, future in futures:
            vector, usage, batch = await asyncio.shield(future)
            vectors[text] = vector
            if batch not in counted_batches:
                counted_batches.add(batch)
                requests += usage.requests
                input_tokens = _sum_optional(input_tokens, usage.input_tokens)
                output_tokens = _sum_optional(output_tokens, usage.output_tokens)
        ordered = tuple(vectors[text] for text in texts)
        return EmbeddingResult(
            vectors=ordered,
            usage=InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def _ensure_loop_resources(self) -> None:
        if self._lock is not None:
            return
        self._lock = asyncio.Lock()
        self._work = asyncio.Event()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="powercontext-embedding-coalescing")

    async def _dispatch_loop(self) -> None:
        while True:
            await self._work.wait()
            self._work.clear()
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            while self._pending:
                batch = self._pending[: self._max_batch_size]
                del self._pending[: self._max_batch_size]
                self._batch_sequence += 1
                await self._run_batch(tuple(batch), self._batch_sequence)

    async def _run_batch(self, batch: tuple[str, ...], sequence: int) -> None:
        try:
            result = await self._model.embed(batch)
            if len(result.vectors) != len(batch):
                message = "embedding provider returned a mismatching vector count"
                raise ValueError(message)
            usage = result.usage
            for text, vector in zip(batch, result.vectors, strict=True):
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                future = self._waiters.pop(text, None)
                if future is not None and not future.done():
                    future.set_result((vector, usage, sequence))
        except BaseException as error:  # noqa: BLE001 - waiters must observe every failure
            for text in batch:
                future = self._waiters.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)


def _sum_optional(current: int | None, addition: int | None) -> int | None:
    if addition is None:
        return current
    return addition if current is None else current + addition


__all__ = ["CoalescingEmbeddingModel"]
