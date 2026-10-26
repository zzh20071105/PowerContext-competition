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

"""Full physical-page PDF capture with exact page-level evidence citations."""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

PDF_PAGE_SOURCE_NAME = "pdf-page"
_MAX_SOURCE_ID_CHARS = MAX_SOURCE_ID_LENGTH


class PDFDocumentInput(BaseModel):
    """One complete original PDF document addressed by a stable document identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page's extractable text bound to its document revision digest."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured original PDF page kept as immutable revision-scoped evidence."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable page identity ``pdf:{document}:{sha256}:{page}``."""

    source_id = (
        f"pdf:{quote(document_id, safe='')}:{quote(document_sha256, safe='')}:{page_number}"
    )
    if len(source_id) > _MAX_SOURCE_ID_CHARS:
        message = "pdf page source_id must not exceed 256 characters"
        raise ValueError(message)
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page in order, failing atomically on broken documents."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        message = "document_id must be a non-empty string that is not whitespace-only"
        raise ValueError(message)
    if not value.content:
        message = "PDF content must not be empty"
        raise ValueError(message)
    try:
        reader = PdfReader(io.BytesIO(value.content))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                message = "encrypted PDF requires a user password"
                raise ValueError(message)
        pages = tuple(reader.pages)
        texts = tuple(page.extract_text() or "" for page in pages)
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - every pypdf failure is one atomic ValueError
        message = "PDF content could not be parsed completely"
        raise ValueError(message) from error
    document_sha256 = hashlib.sha256(value.content).hexdigest()
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=document_sha256,
            page_number=number,
            text=texts[number - 1],
        )
        for number in range(1, len(pages) + 1)
    )


class PDFPageAdapter:
    """Resolve and read captured PDF pages under the ``pdf-page`` Definition."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Project one captured page as document and page text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory for the question and return distinct page citations."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    if limit < 1:
        message = "limit must be a positive integer"
        raise ValueError(message)
    citations: list[MemoryCitation] = []
    seen_sources: set[str] = set()
    search_limit = max(limit * 2, limit)
    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=search_limit))
    for hit in page.hits:
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        identities = tuple(reference.source_id for reference in entry.sources)
        if not identities or any(identity in seen_sources for identity in identities):
            continue
        seen_sources.update(identities)
        citations.append(citation)
        if len(citations) == limit:
            break
    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
