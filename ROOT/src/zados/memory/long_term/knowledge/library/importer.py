"""
Library Importer — reads .txt files and ingests them into LibraryStore.

Supports two strategies:
  - "whole"   : one LibraryEntry per file (small docs, articles)
  - "chunked" : split on paragraph/section breaks, one entry per chunk
                (books, long documents)

Chunked entries share a group_id so they can be reassembled if needed.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zados.memory.long_term.knowledge.library.store import LibraryStore
from zados.memory.long_term.knowledge.types import LibraryEntry

log = logging.getLogger(__name__)

# Max characters per chunk (≈ 1500 words).  TF-IDF works best on
# moderate-sized documents — too large and every term appears, too
# small and there's no discriminative power.
DEFAULT_CHUNK_SIZE = 6000
MIN_CHUNK_SIZE = 200


@dataclass
class ImportResult:
    """Summary of a file import operation."""
    file_path: str = ""
    title: str = ""
    strategy: str = ""
    entries_created: int = 0
    total_chars: int = 0
    group_id: str = ""
    entry_ids: List[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SECTION_BREAK = re.compile(
    r"\n\s*\n"          # double newline (paragraph break)
    r"|\n-{3,}\n"       # horizontal rule (---)
    r"|\n={3,}\n"       # horizontal rule (===)
    r"|\n#{1,6}\s"      # markdown heading
    r"|\nChapter\s+\d"  # "Chapter N" heading
    r"|\n\d+\.\s+[A-Z]" # numbered section ("1. Introduction")
)


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    min_chars: int = MIN_CHUNK_SIZE,
) -> List[str]:
    """Split text into chunks on natural section/paragraph boundaries.

    Parameters
    ----------
    text : str
        Full document text.
    max_chars : int
        Target maximum characters per chunk.
    min_chars : int
        Minimum chunk size; smaller fragments are merged into the previous.

    Returns
    -------
    list[str]
        Non-empty chunks.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    # Split on section breaks
    parts = _SECTION_BREAK.split(text)

    chunks: List[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(current) + len(part) + 1 <= max_chars:
            current = (current + "\n\n" + part).strip() if current else part
        else:
            if current:
                chunks.append(current)
            # If a single part exceeds max_chars, force-split on sentences
            if len(part) > max_chars:
                sub_chunks = _force_split(part, max_chars)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Merge tiny trailing chunks
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)

    return merged


def _force_split(text: str, max_chars: int) -> List[str]:
    """Hard-split on sentence boundaries when a section is too large."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

def import_file(
    store: LibraryStore,
    file_path: str,
    title: str = "",
    domain: str = "",
    tags: Optional[List[str]] = None,
    source_type: str = "book",
    strategy: str = "auto",
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    nt_snapshot: Optional[Dict[str, float]] = None,
) -> ImportResult:
    """Read a .txt file and ingest it into the LibraryStore.

    Parameters
    ----------
    store : LibraryStore
        Target store.
    file_path : str
        Absolute path to a .txt file (UTF-8).
    title : str
        Document title; defaults to filename stem.
    domain : str
        Knowledge domain (e.g. "biology", "philosophy").
    tags : list[str], optional
        Searchable tags.
    source_type : str
        "book" | "article" | "document" | "upload"
    strategy : str
        "whole" — one entry per file.
        "chunked" — split on paragraph/section breaks.
        "auto" — chunked if > max_chunk_chars, whole otherwise.
    max_chunk_chars : int
        Target max characters per chunk (only for chunked strategy).
    nt_snapshot : dict, optional
        Current NT concentrations to stamp on entries.

    Returns
    -------
    ImportResult
    """
    result = ImportResult(file_path=file_path)
    tags = tags or []

    # --- Read file ---
    if not os.path.isfile(file_path):
        result.error = f"File not found: {file_path}"
        log.error(result.error)
        return result

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as exc:
        result.error = f"Failed to read file: {exc}"
        log.error(result.error)
        return result

    if not text.strip():
        result.error = "File is empty."
        return result

    result.total_chars = len(text)
    result.title = title or os.path.splitext(os.path.basename(file_path))[0]

    # --- Decide strategy ---
    if strategy == "auto":
        strategy = "chunked" if len(text) > max_chunk_chars else "whole"
    result.strategy = strategy

    group_id = str(uuid.uuid4())
    result.group_id = group_id

    # --- Ingest ---
    if strategy == "whole":
        entry = store.ingest(
            title=result.title,
            content=text,
            source_type=source_type,
            domain=domain,
            tags=tags + ["group:" + group_id],
            nt_snapshot=nt_snapshot or {},
        )
        result.entries_created = 1
        result.entry_ids.append(entry.entry_id)
    else:
        chunks = chunk_text(text, max_chars=max_chunk_chars)
        for i, chunk in enumerate(chunks):
            chunk_title = f"{result.title} [{i + 1}/{len(chunks)}]"
            entry = store.ingest(
                title=chunk_title,
                content=chunk,
                source_type=source_type,
                domain=domain,
                tags=tags + [
                    "group:" + group_id,
                    f"chunk:{i + 1}/{len(chunks)}",
                ],
                nt_snapshot=nt_snapshot or {},
            )
            result.entry_ids.append(entry.entry_id)
        result.entries_created = len(chunks)

    log.info(
        "Library import: '%s' — %s strategy, %d entries, %d chars",
        result.title, result.strategy, result.entries_created, result.total_chars,
    )
    return result


def import_text(
    store: LibraryStore,
    title: str,
    content: str,
    domain: str = "",
    tags: Optional[List[str]] = None,
    source_type: str = "document",
    strategy: str = "auto",
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    nt_snapshot: Optional[Dict[str, float]] = None,
) -> ImportResult:
    """Ingest raw text (no file) into the LibraryStore.

    Same as import_file but accepts a string directly.
    """
    result = ImportResult(title=title, total_chars=len(content))
    tags = tags or []

    if not content.strip():
        result.error = "Content is empty."
        return result

    if strategy == "auto":
        strategy = "chunked" if len(content) > max_chunk_chars else "whole"
    result.strategy = strategy

    group_id = str(uuid.uuid4())
    result.group_id = group_id

    if strategy == "whole":
        entry = store.ingest(
            title=title,
            content=content,
            source_type=source_type,
            domain=domain,
            tags=tags + ["group:" + group_id],
            nt_snapshot=nt_snapshot or {},
        )
        result.entries_created = 1
        result.entry_ids.append(entry.entry_id)
    else:
        chunks = chunk_text(content, max_chars=max_chunk_chars)
        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} [{i + 1}/{len(chunks)}]"
            entry = store.ingest(
                title=chunk_title,
                content=chunk,
                source_type=source_type,
                domain=domain,
                tags=tags + [
                    "group:" + group_id,
                    f"chunk:{i + 1}/{len(chunks)}",
                ],
                nt_snapshot=nt_snapshot or {},
            )
            result.entry_ids.append(entry.entry_id)
        result.entries_created = len(chunks)

    log.info(
        "Library text import: '%s' — %s strategy, %d entries, %d chars",
        title, result.strategy, result.entries_created, result.total_chars,
    )
    return result
