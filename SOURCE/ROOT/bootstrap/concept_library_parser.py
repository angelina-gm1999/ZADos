"""
concept_library_parser.py
=========================

Parse the ZA-DOS concept library document (Layers 1.1–3.5) into structured
ConceptEntry objects.

The document uses a repeating block schema separated by ``---`` lines:

    CONCEPT:          <name>
    LAYER:            <code>
    ALIASES:          <comma-separated>
    DEFINITION:       <multiline text>
    DEPENDS-ON:       <comma-separated>
    ATOM-LINKS:
      <LinkType>  → <target>   (<optional note>)
    CONCEPTUAL-SCOPE: <multiline text>
    REWARD-DOMAIN:    <comma-separated>
    ENGINE-RELEVANCE: <comma-separated>
    SOURCES:          <text>
    TV-SEED:          HIGH / MEDIUM / LOW
    FLAGS:            <optional>

Public API
----------
    parse_concept_library(path) -> List[ConceptEntry]
    get_default_library_path()  -> str
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AtomLinkSpec:
    link_type: str   # e.g. "InheritanceLink"
    target: str      # cleaned target concept name
    note: str = ""


@dataclass
class ConceptEntry:
    name: str
    layer: str               # e.g. "1.1"
    layer_group: str         # integer part before the dot, e.g. "1"
    aliases: List[str] = field(default_factory=list)
    definition: str = ""
    depends_on: List[str] = field(default_factory=list)
    atom_links: List[AtomLinkSpec] = field(default_factory=list)
    conceptual_scope: str = ""
    reward_domains: List[str] = field(default_factory=list)
    engine_relevance: List[str] = field(default_factory=list)
    sources: str = ""
    tv_seed: str = "MEDIUM"  # HIGH / MEDIUM / LOW
    flags: str = ""


# ---------------------------------------------------------------------------
# Known field keywords (used to detect field boundaries)
# ---------------------------------------------------------------------------

_FIELD_KEYWORDS = {
    "CONCEPT",
    "LAYER",
    "ALIASES",
    "DEFINITION",
    "DEPENDS-ON",
    "ATOM-LINKS",
    "CONCEPTUAL-SCOPE",
    "REWARD-DOMAIN",
    "ENGINE-RELEVANCE",
    "SOURCES",
    "TV-SEED",
    "FLAGS",
}

_FIELD_RE = re.compile(
    r"^("
    + "|".join(re.escape(k) for k in sorted(_FIELD_KEYWORDS, key=len, reverse=True))
    + r")\s*:\s*(.*)",
)


# ---------------------------------------------------------------------------
# ATOM-LINKS parsing helpers
# ---------------------------------------------------------------------------

_ATOM_ARROW_RE = re.compile(r"→")
_PARENS_RE = re.compile(r"\(.*?\)")
_BRACKETS_RE = re.compile(r"^\[(.+)\]$")

# Only these are valid AtomSpace link types that we recognise
_VALID_LINK_TYPES = frozenset({
    "InheritanceLink",
    "SimilarityLink",
    "EvaluationLink",
    "ImplicationLink",
    "HebbianLink",
    "NotLink",
    "ListLink",
    "AndLink",
    "OrLink",
})


def _parse_atom_link_line(line: str) -> AtomLinkSpec | None:
    """Parse a single ATOM-LINKS line containing '→'.

    Example inputs:
        "  InheritanceLink  → exists  (root concept)"
        "  SimilarityLink   → [thing, property]   (note)"
    """
    if "→" not in line:
        return None

    parts = line.split("→", 1)
    link_type_raw = parts[0].strip()
    right = parts[1].strip()

    # Remove parenthetical notes to get target
    note_match = _PARENS_RE.search(right)
    note = note_match.group(0).strip("()").strip() if note_match else ""
    target_raw = _PARENS_RE.sub("", right).strip()

    # Normalise target: strip brackets, take first element of list
    bracket_match = _BRACKETS_RE.match(target_raw)
    if bracket_match:
        items = [t.strip() for t in bracket_match.group(1).split(",")]
        target = items[0] if items else target_raw
    else:
        # Take up to the first comma (handles trailing notes)
        target = target_raw.split(",")[0].strip()

    # Clean up target — remove any trailing whitespace / extra words after space
    # (some targets are hyphenated phrases; keep them whole but trim whitespace)
    target = target.strip()

    if not link_type_raw or not target:
        return None

    # Skip non-standard or unrecognised link types
    if link_type_raw not in _VALID_LINK_TYPES:
        return None

    return AtomLinkSpec(link_type=link_type_raw, target=target, note=note)


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------

def _extract_layer_group(layer: str) -> str:
    """Return the integer part of the layer string, e.g. '1.1' → '1'."""
    if not layer:
        return ""
    return layer.split(".")[0].strip()


def _split_csv(text: str) -> List[str]:
    """Split comma-separated text, strip each item, drop empty strings."""
    return [s.strip() for s in text.split(",") if s.strip()]


def _parse_block(lines: List[str]) -> ConceptEntry | None:
    """Parse a list of lines (one concept block) into a ConceptEntry.

    Returns None if the block has no CONCEPT field.
    """
    # Collect field → raw text
    fields: dict[str, list[str]] = {}
    current_field: str | None = None

    for line in lines:
        m = _FIELD_RE.match(line)
        if m:
            current_field = m.group(1)
            rest = m.group(2)
            fields.setdefault(current_field, [])
            if rest:
                fields[current_field].append(rest)
        else:
            if current_field is not None:
                fields[current_field].append(line.rstrip())

    name = " ".join(fields.get("CONCEPT", [])).strip()
    if not name:
        return None

    layer = " ".join(fields.get("LAYER", [])).strip()
    layer_group = _extract_layer_group(layer)

    # Aliases — join and split by comma
    aliases_raw = " ".join(fields.get("ALIASES", [])).strip()
    aliases = _split_csv(aliases_raw)

    # Definition — join lines, strip excessive whitespace
    definition_lines = fields.get("DEFINITION", [])
    definition = " ".join(l.strip() for l in definition_lines if l.strip())

    # DEPENDS-ON
    depends_raw = " ".join(fields.get("DEPENDS-ON", [])).strip()
    if "[none" in depends_raw.lower() or depends_raw.lower().startswith("[none"):
        depends_on: List[str] = []
    else:
        # Strip surrounding brackets if present
        depends_raw_clean = depends_raw.strip("[]")
        depends_on = _split_csv(depends_raw_clean)

    # ATOM-LINKS — collect only lines with →
    atom_link_lines = fields.get("ATOM-LINKS", [])
    atom_links: List[AtomLinkSpec] = []
    for al_line in atom_link_lines:
        spec = _parse_atom_link_line(al_line)
        if spec is not None:
            atom_links.append(spec)

    # CONCEPTUAL-SCOPE
    scope_lines = fields.get("CONCEPTUAL-SCOPE", [])
    conceptual_scope = " ".join(l.strip() for l in scope_lines if l.strip())

    # REWARD-DOMAIN
    reward_raw = " ".join(fields.get("REWARD-DOMAIN", [])).strip()
    reward_domains = _split_csv(reward_raw)

    # ENGINE-RELEVANCE
    engine_raw = " ".join(fields.get("ENGINE-RELEVANCE", [])).strip()
    engine_relevance = _split_csv(engine_raw)

    # SOURCES
    sources_lines = fields.get("SOURCES", [])
    sources = " ".join(l.strip() for l in sources_lines if l.strip())

    # TV-SEED — take first token
    tv_raw = " ".join(fields.get("TV-SEED", [])).strip()
    tv_token = tv_raw.split()[0].upper() if tv_raw else "MEDIUM"
    tv_seed = tv_token if tv_token in {"HIGH", "MEDIUM", "LOW"} else "MEDIUM"

    # FLAGS
    flags_lines = fields.get("FLAGS", [])
    flags = " ".join(l.strip() for l in flags_lines if l.strip())

    return ConceptEntry(
        name=name,
        layer=layer,
        layer_group=layer_group,
        aliases=aliases,
        definition=definition,
        depends_on=depends_on,
        atom_links=atom_links,
        conceptual_scope=conceptual_scope,
        reward_domains=reward_domains,
        engine_relevance=engine_relevance,
        sources=sources,
        tv_seed=tv_seed,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_concept_library(path: str) -> List[ConceptEntry]:
    """Parse the full ZA-DOS concept library document.

    Parameters
    ----------
    path : str
        Absolute path to the concept library text file.

    Returns
    -------
    List[ConceptEntry]
        All parsed concept entries (blocks without a CONCEPT field are skipped).
    """
    with open(path, encoding="utf-8") as fh:
        raw_text = fh.read()

    # Split on '---' separator lines to get raw blocks
    separator_re = re.compile(r"^-{3,}\s*$", re.MULTILINE)
    raw_blocks = separator_re.split(raw_text)

    # Some blocks may start with a new CONCEPT: line without a --- separator.
    # Re-split each block on lines that look like "CONCEPT: ..." at column 0.
    concept_start_re = re.compile(r"^(?=CONCEPT\s*:)", re.MULTILINE)
    all_raw_line_groups: List[List[str]] = []
    for block in raw_blocks:
        sub_blocks = concept_start_re.split(block)
        for sub in sub_blocks:
            lines = sub.splitlines()
            all_raw_line_groups.append(lines)

    entries: List[ConceptEntry] = []
    seen_names: set[str] = set()

    for line_group in all_raw_line_groups:
        entry = _parse_block(line_group)
        if entry is None:
            continue
        # Skip schema/example entries (from the header section)
        if "lowercase" in entry.name or "canonical name" in entry.name.lower():
            continue
        # Skip entries with no layer (likely header text)
        if not entry.layer:
            continue
        # Deduplicate by name (case-insensitive)
        key = entry.name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        entries.append(entry)

    return entries


def get_default_library_path() -> str:
    """Return the canonical path to the concept library.

    Navigates from this file's location up to ROOT, then into
    knowledge_sources/books/.
    """
    # This file lives at: ROOT/src/zados/bootstrap/concept_library_parser.py
    bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
    # bootstrap_dir → zados/bootstrap
    zados_dir = os.path.dirname(bootstrap_dir)          # zados/
    src_dir = os.path.dirname(zados_dir)                # src/
    root_dir = os.path.dirname(src_dir)                 # ROOT/
    return os.path.join(
        root_dir, "knowledge_sources", "books",
        "zadOS_concept_library_COMPLETE.txt",
    )
