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
import math
import re
from collections import Counter
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

# Question words that describe asking, not the asked-about content. Retrieval
# ranks pages by the remaining content terms only.
_STOP_WORDS = frozenset((
    "a", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "being", "both",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "if", "in", "into", "is", "it", "its", "many", "may", "much", "no", "not", "of",
    "on", "or", "please", "should", "so", "such", "than", "that", "the", "their", "then",
    "there", "these", "they", "this", "those", "to", "too", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "company", "fiscal",
    "year", "fy", "10k", "10q", "between", "during", "about", "per", "each", "more", "most",
    "other", "some", "own", "same", "only", "just", "based", "respectively", "within",
    "under", "over", "upon", "among", "across", "after", "before", "because", "due",
    "using", "used", "use", "useful", "metric", "report", "reported", "reports", "give",
    "given", "according", "following", "follow", "like", "state",
))

# Canonical report-section signatures. FinanceBench-style questions name analyst
# concepts (quick ratio, capex, legal battles) whose words never appear on the
# annotated statement pages, so one ranking signal expands each question with the
# section phrases that identify those pages inside SEC filings. Filings that name
# their sections differently are covered by the statistical signal fused below.
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


def _stem(word: str) -> str:
    """Collapse plural inflection so ``customer``/``customers`` or
    ``revenue``/``revenues`` match; the same function runs on questions and
    pages, so consistent collisions are harmless."""

    if len(word) <= 3:
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _content_terms(text: str) -> list[str]:
    """Stemmed content terms of a question: numbers and words above noise length."""

    return [
        stem
        for stem in (_stem(word) for word in _WORD.findall(text.lower()))
        if (len(stem) > 2 or stem.isdigit()) and stem not in _STOP_WORDS
    ]


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
    """Derive citation, document and stemmed word counts from one hit without any get.

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
        Counter(_stem(word) for word in _WORD.findall(body.lower())),
    )


async def _sweep(memory: Any, query: str, fetch: int) -> list[tuple[MemoryCitation, str, Counter]]:
    result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=fetch))
    resolved = []
    for hit in result.hits:
        info = _hit_info(hit)
        if info is not None:
            resolved.append(info)
    return resolved


class _PageTable:
    """Page-grain term statistics for every captured page in one scoped Memory.

    ``counts``/``tokens`` stem with the same :func:`_stem` as questions, ``idf``
    is computed over the page population of the whole scope, and ``entry_keys``
    maps every journal entry to its page so probe hits resolve without a get.
    """

    def __init__(self, listed: Any) -> None:
        self.counts: dict[tuple[str, int], Counter] = {}
        self.tokens: dict[tuple[str, int], list[str]] = {}
        self.citations: dict[tuple[str, int], MemoryCitation] = {}
        self.entry_keys: dict[str, tuple[str, int]] = {}
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
            text = record.entry.text
            body = text.partition("\n")[2] if text.startswith(document_id + "\n") else text
            words = [_stem(word) for word in _WORD.findall(body.lower())]
            self.counts.setdefault(key, Counter()).update(words)
            self.tokens.setdefault(key, []).extend(words)
            self.citations.setdefault(key, record.citation)
            self.entry_keys[record.citation.entry_id] = key

    def document_names(self) -> set[str]:
        return {key[0] for key in self.counts}

    def idf(self) -> dict[str, float]:
        """Inverse document frequency over the scope's page population."""

        total = max(1, len(self.counts))
        document_frequency: Counter[str] = Counter()
        for counts in self.counts.values():
            document_frequency.update(counts.keys())
        return {term: math.log(1 + total / df) for term, df in document_frequency.items()}

    def document_frequency(self, term: str) -> int:
        return sum(1 for counts in self.counts.values() if term in counts)


def _anchor_document(
    probe: list[tuple[MemoryCitation, str, Counter]],
    question: str,
    documents: set[str],
) -> str:
    """Pick the known document that the question names outright.

    A question names its company either as a whole token ("3M", "PepsiCo") or as
    run-together words ("American Water Works" vs ``AMERICANWATERWORKS``); only
    real document ids from the listing are considered.
    """

    words = set(_WORD.findall(question.lower()))
    joined = "".join(_WORD.findall(question.lower()))
    probe_docs = [document_id for _citation, document_id, _counts in probe]
    candidates = list(dict.fromkeys([*probe_docs, *sorted(documents)]))
    for document_id in candidates:
        if document_id not in documents:
            continue
        match = re.search(r"[a-z0-9]+", document_id.lower())
        company = match.group(0) if match else ""
        if not company:
            continue
        if company in words or (len(company) >= 8 and company in joined):
            return document_id
    for document_id in probe_docs:
        if document_id in documents:
            return document_id
    return next(iter(sorted(documents)), "")


def _section_plan(question: str) -> list[str]:
    lowered = question.lower()
    sections = list(_ALWAYS_SECTIONS)
    for name, triggers in _SECTION_TRIGGERS:
        if any(trigger in lowered for trigger in triggers) and name not in sections:
            sections.append(name)
    return sections


_HEAD_TOKENS = 40
_SECTION_HEAD_WEIGHT = 5
_TAIL_TF_CAP = 3


def _score_section(
    counts: Counter,
    head_set: set[str],
    phrases: tuple[str, ...],
    tokens: tuple[str, ...],
) -> int:
    total = 0
    for phrase in phrases:
        phrase_tokens = [_stem(token) for token in phrase.split()]
        if all(counts.get(token, 0) for token in phrase_tokens):
            # A section names itself in its leading lines; the same words
            # mentioned mid-page are only a discussion of that section.
            total += _SECTION_HEAD_WEIGHT if all(token in head_set for token in phrase_tokens) else 3
    total += sum(min(counts.get(_stem(token), 0), 2) for token in tokens)
    return total


def _signature_ranked_keys(
    table: _PageTable,
    document_id: str,
    question: str,
    limit: int,
) -> list[tuple[str, int]]:
    """Section-signature ranking at page grain, mirroring an analyst's map of
    the filing: statements, MD&A, segments, risk, legal, tax, plus a
    question-overlap tail for concepts the signatures do not cover."""

    document_pages = [key for key in table.counts if key[0] == document_id]
    if not document_pages:
        return []
    ordered: list[tuple[str, int]] = []
    taken: set[tuple[str, int]] = set()

    def push(ranked: list[tuple[str, int]], count: int) -> None:
        for key in ranked:
            if count <= 0 or len(ordered) >= limit:
                return
            if key in taken:
                continue
            taken.add(key)
            ordered.append(key)
            count -= 1

    lowered = question.lower()
    for name, quota, phrases, tokens in (s for s in _SECTION_SIGNATURES if s[0] in _section_plan(question)):
        ranked = sorted(
            document_pages,
            key=lambda key: (
                -_score_section(table.counts[key], set(table.tokens[key][:_HEAD_TOKENS]), phrases, tokens),
                key[0],
                key[1],
            ),
        )
        push(ranked, quota)
        triggers = dict(_SECTION_TRIGGERS).get(name, ())
        if any(trigger in lowered for trigger in triggers):
            push(ranked, 2)
    question_words = {
        stem
        for stem in (_stem(word) for word in _WORD.findall(lowered))
        if len(stem) > 2 and stem not in _STOP_WORDS and not stem.isdigit()
    }
    # Raw counts let one repeated year on a table page outweigh every content
    # word; the cap keeps the tail a question-overlap signal.
    ranked_question = sorted(
        document_pages,
        key=lambda key: (
            -sum(min(table.counts[key].get(word, 0), _TAIL_TF_CAP) for word in question_words),
            key[0],
            key[1],
        ),
    )
    push(ranked_question, max(0, limit - len(ordered)))
    return ordered


def _pseudo_pages(
    table: _PageTable,
    probe: list[tuple[MemoryCitation, str, Counter]],
    document_id: str,
) -> list[tuple[str, int]]:
    """Distinct pages behind the probe hits, target document's pages first."""

    inside: list[tuple[str, int]] = []
    outside: list[tuple[str, int]] = []
    for citation, _document, _counts in probe:
        key = table.entry_keys.get(citation.entry_id)
        if key is None or key in inside or key in outside:
            continue
        if key[0] == document_id:
            inside.append(key)
        else:
            outside.append(key)
    return (inside + outside)[:4]


def _weighted_question(
    table: _PageTable,
    idf: dict[str, float],
    question: str,
    question_terms: list[str],
    pseudo_pages: list[tuple[str, int]],
) -> dict[str, float]:
    """Question terms plus two faithful expansions of the filing's own text.

    Fiscal-year mentions ("FY2022") are normalized to the bare year that pages
    actually print, and the pages the full-text probe did find contribute their
    rare vocabulary: analyst concepts ("quick ratio", "customer concentration")
    often never appear verbatim on the pages that answer them, but the probe's
    own pages carry the filing's wording around the concept.
    """

    weighted = dict.fromkeys(question_terms, 1.0)
    lowered = question.lower()
    for year in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", lowered):
        if year not in weighted:
            weighted[year] = 0.5
    if pseudo_pages:
        question_set = set(question_terms)
        total = max(1, len(table.counts))
        feedback: Counter[str] = Counter()
        for key in pseudo_pages:
            feedback.update(table.counts.get(key, ()))
        ranked = sorted(
            (
                (tf * idf.get(term, 0.0), term)
                for term, tf in feedback.items()
                if term not in question_set
                and len(term) > 2
                and table.document_frequency(term) <= total // 4
            ),
            reverse=True,
        )[:8]
        for _score, term in ranked:
            weighted[term] = 0.6
    return weighted


def _bm25(
    counts: Counter,
    length: int,
    average_length: float,
    idf: dict[str, float],
    weighted_terms: dict[str, float],
) -> float:
    if length <= 0 or not weighted_terms:
        return 0.0
    k1, b = 1.2, 0.75
    normalizer = k1 * (1 - b + b * length / max(1.0, average_length))
    score = 0.0
    for term, weight in weighted_terms.items():
        tf = counts.get(term, 0)
        if tf:
            score += weight * idf.get(term, 0.0) * tf * (k1 + 1) / (tf + normalizer)
    return score


def _phrase_bonus(
    tokens: list[str],
    head_size: int,
    question_terms: list[str],
    idf: dict[str, float],
) -> float:
    """Reward pages where two consecutive question words also occur adjacently.

    A page containing the query's exact word order ("net income", "cash flow")
    is a far better answer candidate than the same words scattered apart, and a
    match in the page's leading lines weighs double: statements, notes and
    section openers carry their titles in their first lines.
    """

    if len(tokens) < 2 or len(question_terms) < 2:
        return 0.0
    bigrams = {
        (question_terms[index], question_terms[index + 1])
        for index in range(len(question_terms) - 1)
    }
    bonus = 0.0
    for index in range(len(tokens) - 1):
        pair = (tokens[index], tokens[index + 1])
        if pair in bigrams:
            weight = 1.0 if index < head_size - 1 else 0.5
            bonus += weight * (idf.get(pair[0], 0.0) + idf.get(pair[1], 0.0))
    return bonus


def _bm25_ranked_keys(
    table: _PageTable,
    idf: dict[str, float],
    document_id: str,
    question: str,
    question_terms: list[str],
    probe: list[tuple[MemoryCitation, str, Counter]],
    limit: int,
) -> list[tuple[str, int]]:
    """Statistical ranking of one filing's pages.

    BM25 over stemmed page statistics with inverse document frequency from the
    page population, adjacency bonuses for the query's word order, and feedback
    terms from the probe's own pages. No hand-written vocabulary decides alone;
    the section-signature ranking is fused with this one by the caller.
    """

    document_pages = sorted(key for key in table.counts if key[0] == document_id)
    if not document_pages or not question_terms:
        return []
    lengths = {key: sum(table.counts[key].values()) for key in document_pages}
    average_length = sum(lengths.values()) / len(document_pages)
    weighted_terms = _weighted_question(
        table, idf, question, question_terms, _pseudo_pages(table, probe, document_id)
    )
    raw = [
        _bm25(table.counts[key], lengths[key], average_length, idf, weighted_terms)
        + _phrase_bonus(table.tokens[key], _HEAD_TOKENS, question_terms, idf)
        for key in document_pages
    ]
    # Evidence pages come in contiguous runs - a statement spans several pages,
    # MD&A tables continue overleaf - so every page inherits part of the best
    # score in its immediate neighborhood before the ranking is cut.
    smoothed = [
        0.65 * raw[index]
        + 0.35 * max(raw[neighbor] for neighbor in range(max(0, index - 2), min(len(raw), index + 3)))
        for index in range(len(raw))
    ]
    scored = sorted(
        ((-smoothed[index], key[0], key[1]) for index, key in enumerate(document_pages)),
    )
    return [(key[1], key[2]) for key in scored[:limit]]


def _fuse_rankings(*rankings: list[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    """Reciprocal-rank fusion: pages that several independent rankings agree on
    outrank pages that only one ranking put forward, and a page each ranking
    missed by one signal can still be carried by the other."""

    scores: dict[tuple[str, int], float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))[:limit]


async def _select_page_citations(
    memory: Any,
    question: str,
    limit: int,
) -> list[MemoryCitation]:
    """Anchor the target document, fuse the section and statistical rankings,
    and return distinct page citations in fused-rank order."""

    # 1. One public listing resolves every entry's page identity and the page-
    #    grain term statistics for the whole capture.
    listed = await memory.list(include_inactive=True)
    table = _PageTable(listed)
    idf = table.idf()

    # 2. Question probe: the full-text ranking over entry parts anchors the
    #    target document and supplies the feedback pages.
    probe = await _sweep(memory, question, 32)
    target_doc = _anchor_document(probe, question, table.document_names())
    question_terms = _content_terms(question)

    # 3. Two independent page rankings over the target filing, fused; both
    #    rankings contribute their deeper lists, only the top ``limit`` pages
    #    are emitted. The probe-order tail catches anything both rankings
    #    could still have missed.
    fused = _fuse_rankings(
        _signature_ranked_keys(table, target_doc, question, 30),
        _bm25_ranked_keys(table, idf, target_doc, question, question_terms, probe, 30),
        limit=limit,
    )
    emitted: list[MemoryCitation] = []
    emitted_keys: set[tuple[str, int]] = set()

    def take(citation: MemoryCitation) -> None:
        key = table.entry_keys.get(citation.entry_id)
        if key is None or key in emitted_keys or len(emitted) >= limit:
            return
        emitted_keys.add(key)
        emitted.append(citation)

    for key in fused:
        if len(emitted) >= limit:
            break
        take(table.citations[key])
    for citation, document_id, _counts in probe:
        if len(emitted) >= limit:
            break
        if document_id == target_doc:
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
