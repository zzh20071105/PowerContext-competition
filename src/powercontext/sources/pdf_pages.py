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
import re
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

PDF_PAGE_SOURCE_NAME = "pdf-page"
_MAX_SOURCE_ID_CHARS = MAX_SOURCE_ID_LENGTH

_WORD = re.compile(r"[a-z0-9]+")

# Canonical report-section signatures. FinanceBench-style questions name analyst
# concepts (quick ratio, capex, legal battles) whose words never appear on the
# annotated statement pages, so retrieval expands each question with the section
# phrases that identify those pages inside SEC filings.
_SECTION_SIGNATURES: tuple[tuple[str, int, tuple[str, ...], tuple[str, ...]], ...] = (
    ("income", 2,
     ("consolidated statement of income", "net sales", "cost of sales", "operating profit", "per share"),
     ("statement", "income", "net", "sales", "cost", "gross", "operating", "profit", "share")),
    ("balance", 2,
     ("consolidated balance sheet", "total assets", "total current assets", "stockholders equity"),
     ("balance", "sheet", "total", "assets", "liabilities", "equity", "current")),
    ("cashflow", 3,
     ("consolidated statement of cash flows", "operating activities", "investing activities",
      "purchases of property"),
     ("statement", "cash", "flows", "operating", "investing", "financing", "activities",
      "depreciation", "capital")),
    ("business", 3,
     ("reportable segments", "business segment", "our operations"),
     ("segment", "divisions", "customers", "products", "operations", "markets")),
    ("mda", 2,
     ("management's discussion and analysis", "results of operations"),
     ("management", "discussion", "analysis", "results", "operations", "overview")),
    ("risk", 2,
     ("forward-looking statements", "risk factors"),
     ("forward", "looking", "risk", "assumptions", "expectations", "could")),
    ("legal", 2,
     ("legal proceedings", "commitments and contingencies"),
     ("legal", "litigation", "proceedings", "claim")),
    ("tax", 2,
     ("income taxes", "effective tax rate"),
     ("income", "taxes", "tax", "provision")),
    ("equity", 2,
     ("statement of stockholders equity", "shares outstanding"),
     ("stockholders", "equity", "shares", "repurchases", "dividends")),
)
_SECTION_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("legal", ("legal", "litigation", "lawsuit", "court")),
    ("tax", ("tax",)),
    ("equity", ("buyback", "repurchase", "dividend", "shares outstanding")),
    ("business", ("customer", "product", "geograph", "segment", "cyclicality", "competitor", "business")),
    ("mda", ("increased", "decreased", "expense", "acquisition", "spending")),
    ("risk", ("cyclicality", "competition", "supply", "risk")),
)
_ALWAYS_SECTIONS = ("income", "balance", "cashflow", "business", "mda", "risk")


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
        if reader.is_encrypted and not reader.decrypt(""):
            message = "encrypted PDF requires a user password"
            raise ValueError(message)  # noqa: TRY301 - re-raised unchanged by the handler below
        pages = tuple(reader.pages)
        texts = tuple(page.extract_text() or "" for page in pages)
    except ValueError:
        raise
    except Exception as error:
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


def _hit_info(hit: Any) -> tuple[MemoryCitation, str, Counter] | None:
    """Derive citation, document and word counts from one hit without any get.

    Page entries store the document name as their first line followed by the page
    text, so the document name comes from that line and word counts from the same
    string. Page numbers are resolved lazily, only for entries this module selects.
    """

    if not hit.text:
        return None
    document_id, _, body = hit.text.partition("\n")
    return (
        MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        ),
        document_id,
        Counter(_WORD.findall(body.lower())),
    )


async def _sweep(memory: Any, query: str, fetch: int) -> list[tuple[MemoryCitation, str, Counter]]:
    result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=fetch))
    resolved = []
    for hit in result.hits:
        info = _hit_info(hit)
        if info is not None:
            resolved.append(info)
    return resolved


def _anchor_document(
    probe: list[tuple[MemoryCitation, str, Counter]],
    question: str,
    documents: set[str],
) -> str:
    """Pick the known document whose company token the question names outright.

    Only real document ids from the listing are considered: split entry pieces
    carry arbitrary page text on their first lines, so probe first lines are not
    trustworthy document names.
    """

    question_words = set(_WORD.findall(question.lower()))
    probe_docs = [document_id for _citation, document_id, _counts in probe]
    candidates = list(dict.fromkeys([*probe_docs, *sorted(documents)]))
    for document_id in candidates:
        if document_id not in documents:
            continue
        match = re.search(r"[a-z]+", document_id.lower())
        company = match.group(0) if match else ""
        if company and company in question_words:
            return document_id
    for document_id in probe_docs:
        if document_id in documents:
            return document_id
    return next(iter(sorted(documents)), "")


def _section_plan(question: str) -> list[str]:
    sections = list(_ALWAYS_SECTIONS)
    lowered = question.lower()
    for name, triggers in _SECTION_TRIGGERS:
        if any(trigger in lowered for trigger in triggers) and name not in sections:
            sections.append(name)
    return sections


def _score_section(counts: Counter, phrases: tuple[str, ...], tokens: tuple[str, ...]) -> int:
    total = 0
    for phrase in phrases:
        if all(counts.get(token, 0) for token in phrase.split()):
            total += 3
    total += sum(1 for token in tokens if counts.get(token, 0) >= 2)
    return total


def _rank_pages(
    pages: list[tuple[tuple[str, int], MemoryCitation, Counter]],
    sections: list[str],
    question: str,
    capacity: int,
) -> list[tuple[MemoryCitation, tuple[str, int]]]:
    """Section-signature ranking at page grain, mirroring an analyst's map of
    the filing: statements, MD&A, segments, risk, legal, tax, plus a
    question-overlap tail for concepts the signatures do not cover."""

    merged: list[tuple[MemoryCitation, tuple[str, int]]] = []
    taken_keys: set[tuple[str, int]] = set()

    def push(ranked: list[tuple[tuple[str, int], MemoryCitation, Counter]], count: int) -> None:
        for key, citation, _counts in ranked:
            if count <= 0 or len(merged) >= capacity:
                return
            if key in taken_keys:
                continue
            taken_keys.add(key)
            merged.append((citation, key))
            count -= 1

    for _section, quota, phrases, tokens in (s for s in _SECTION_SIGNATURES if s[0] in sections):
        scorer = partial(_score_section, phrases=phrases, tokens=tokens)
        ranked = sorted(pages, key=lambda info: (-scorer(info), info[0][0], info[0][1]))
        push(ranked, quota)
    question_words = set(_WORD.findall(question.lower()))
    ranked_question = sorted(
        pages, key=lambda info: (-len(question_words & set(info[2])), info[0][0], info[0][1])
    )
    push(ranked_question, max(0, capacity - len(merged)))
    return merged


def _collect_page_entries(listed: Any) -> dict[tuple[str, int], tuple[MemoryCitation, Counter]]:
    """Aggregate listed Memory entries into page-grain token counts."""

    page_entries: dict[tuple[str, int], tuple[MemoryCitation, Counter]] = {}
    for record in listed.entries:
        document_id = ""
        page = 0
        for reference in record.entry.sources:
            parts = reference.source_id.split(":")
            if len(parts) >= 4 and parts[0] == "pdf":
                document_id = parts[1]
                try:
                    page = int(parts[3])
                except ValueError:
                    page = 0
                break
        if not document_id or page < 1:
            continue
        key = (document_id, page)
        existing = page_entries.setdefault(key, (record.citation, Counter()))
        existing[1].update(_WORD.findall(record.entry.text.lower()))
    return page_entries


def _rank_document_pages(
    page_entries: dict[tuple[str, int], tuple[MemoryCitation, Counter]],
    document_id: str,
    question: str,
    limit: int,
) -> list[MemoryCitation]:
    """Section-signature ranking at page grain, mirroring an analyst's map of
    the filing: statements, MD&A, segments, risk, legal, tax, plus a
    question-overlap tail for concepts the signatures do not cover."""

    document_pages = [key for key in page_entries if key[0] == document_id]
    sections = _section_plan(question)
    ordered: list[MemoryCitation] = []
    taken_keys: set[tuple[str, int]] = set()

    def push(ranked: list[tuple[str, int]], count: int) -> None:
        for key in ranked:
            if count <= 0 or len(ordered) >= limit:
                return
            if key in taken_keys:
                continue
            taken_keys.add(key)
            ordered.append(page_entries[key][0])
            count -= 1

    def ranked_for(name: str) -> list[tuple[str, int]]:
        _n, _quota, phrases, tokens = next(s for s in _SECTION_SIGNATURES if s[0] == name)
        return sorted(
            document_pages,
            key=lambda key: (-_score_section(page_entries[key][1], phrases, tokens), key[0], key[1]),
        )

    lowered = question.lower()
    for name, quota, _phrases, _tokens in (s for s in _SECTION_SIGNATURES if s[0] in sections):
        push(ranked_for(name), quota)
        triggers = dict(_SECTION_TRIGGERS).get(name, ())
        if any(trigger in lowered for trigger in triggers):
            push(ranked_for(name), 2)
    stop = {
        "what", "which", "who", "does", "did", "is", "are", "was", "were", "the", "a", "an",
        "of", "in", "for", "to", "and", "or", "by", "as", "on", "at", "any", "its", "their",
        "that", "this", "with", "from", "has", "have", "had", "how", "much", "many", "there",
        "company", "fiscal", "year", "fy", "based", "per", "10k",
    }
    question_words = {
        word for word in _WORD.findall(question.lower()) if len(word) > 2 and word not in stop
    }
    ranked_question = sorted(
        document_pages,
        key=lambda key: (-len(question_words & set(page_entries[key][1])), key[0], key[1]),
    )
    push(ranked_question, max(0, limit - len(ordered)))
    return ordered


async def _select_page_citations(
    memory: Any,
    question: str,
    limit: int,
) -> list[MemoryCitation]:
    """Anchor the target document, score its pages against section signatures,
    and return distinct page citations in emission order."""

    seen: set[str] = set()
    emitted: list[MemoryCitation] = []

    def take(citation: MemoryCitation) -> None:
        if citation.entry_id not in seen:
            seen.add(citation.entry_id)
            emitted.append(citation)

    # 1. One public listing resolves every entry's text and page sources: the
    #    real document set and page-grain token counts for the whole filing.
    listed = await memory.list(include_inactive=True)
    page_entries = _collect_page_entries(listed)

    # 2. Question probe: anchor the target document. Questions name their company
    #    ("3M", "PepsiCo"), so match the document's company token as a whole word;
    #    the raw bm25 winner alone can come from the wrong filing. Keep the
    #    strongest direct match at the head of the result.
    probe = await _sweep(memory, question, 16)
    target_doc = _anchor_document(probe, question, {key[0] for key in page_entries})
    if probe:
        take(probe[0][0])

    # 3. Section-signature ranking at page grain.
    for citation in _rank_document_pages(page_entries, target_doc, question, limit):
        take(citation)

    # 4. Fill any remaining budget with deeper raw-question hits.
    for citation, _document_id, _counts in probe:
        if len(emitted) >= limit:
            break
        take(citation)
    return emitted


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory for the question and return distinct page citations."""

    if limit < 1:
        message = "limit must be a positive integer"
        raise ValueError(message)
    emitted = await _select_page_citations(memory, question, limit)
    return tuple(emitted[:limit])


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
