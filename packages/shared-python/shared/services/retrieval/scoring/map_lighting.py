"""Score-based folding for an oversized ``corpus.outline``/``corpus.node_filter`` map.

Adapted from the archived MAP-NAV pooling/hiding algorithm
(``deprecated/mapnav/nav/nav_map_scores.py``'s ``_pool_unit_scores_to_tree``
and ``nav_projection.py``'s ``_apply_budget_hide``) onto the live,
``document_sections``-based tree ``outline``/``node_filter`` already build —
not onto the retired ``ToolSpace``/``LazyKnowhereProvider`` abstraction those
files used. The deprecated files are unmodified; this is a fresh module using
the same idea against live data.

Only invoked when a rendered map exceeds
``agent_tools.registry.MAP_TOOL_CHAR_BUDGET``. Scoring needs the end user's
original query (``ToolContext.query``) — see that field's docstring for why a
caller without one gets a "narrow the scope" error instead of a silently
unscored fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.models.database.document import (
    DocumentMapUnit,
    DocumentMapUnitIndex,
    DocumentMapUnitToken,
)
from shared.services.retrieval.scoring.knowhere_hybrid import (
    MAP_UNIT_INDEX_FORMAT_VERSION,
    PersistedScoreCorpus,
    PersistedScoreUnit,
    score_persisted_corpus_many,
    tokenize_query_for_ranker,
)
from shared.services.retrieval.scoring.persisted_score_load import (
    build_channel_bm25_stats,
    combine_average_idf,
)

_SCORE_CHANNELS = ("path", "content")


async def load_leaf_unit_scores(
    db: AsyncSession,
    *,
    revision_by_document: dict[str, str],
    section_ids: list[str],
    query: str,
) -> dict[str, float]:
    """BM25 path+content score, pooled onto ``section_id``, for one ``query``.

    Scoped to exactly the ``section_ids`` the caller already rendered (an
    outline/node_filter result), not a document/signal-path predicate — the
    map tools already know their own scope, so this skips rebuilding it in
    SQL. Mirrors ``search.map_unit_discovery``'s scoring formula (same stats
    builder, same channel-sum via ``score_persisted_corpus_many``) but reads
    the persisted map-unit index directly through the ORM instead of the
    discovery module's hand-rolled ``text()`` CTEs, since lighting's scope is
    already a concrete id list, not a filter to compile.

    A section with no map unit at all (never indexed, or the index is stale)
    is simply absent from the returned mapping — callers treat a missing key
    as score ``0.0``, the same as a unit that scored zero.
    """
    query_tokens = tokenize_query_for_ranker(query)
    if not query_tokens or not section_ids or not revision_by_document:
        return {}
    token_hashes = [sha256(token.encode("utf-8")).hexdigest() for token in query_tokens]

    unit_rows = (
        (
            await db.execute(
                select(DocumentMapUnit).where(
                    DocumentMapUnit.section_id.in_(section_ids)
                )
            )
        )
        .scalars()
        .all()
    )
    unit_rows = [
        row
        for row in unit_rows
        if revision_by_document.get(row.document_id) == row.job_result_id
    ]
    if not unit_rows:
        return {}

    map_unit_ids = [row.id for row in unit_rows]
    token_rows = await db.execute(
        select(
            DocumentMapUnitToken.map_unit_id,
            DocumentMapUnitToken.channel,
            DocumentMapUnitToken.token,
            DocumentMapUnitToken.frequency,
        ).where(
            DocumentMapUnitToken.map_unit_id.in_(map_unit_ids),
            DocumentMapUnitToken.channel.in_(_SCORE_CHANNELS),
            DocumentMapUnitToken.token_hash.in_(token_hashes),
        )
    )
    frequencies: dict[tuple[str, str], dict[str, int]] = {}
    for map_unit_id, channel, token, frequency in token_rows.all():
        frequencies.setdefault((str(map_unit_id), str(channel)), {})[str(token)] = int(
            frequency
        )

    revision_pairs = sorted({(row.document_id, row.job_result_id) for row in unit_rows})
    index_rows = (
        (
            await db.execute(
                select(DocumentMapUnitIndex).where(
                    tuple_(
                        DocumentMapUnitIndex.document_id,
                        DocumentMapUnitIndex.job_result_id,
                    ).in_(revision_pairs)
                )
            )
        )
        .scalars()
        .all()
    )
    compatible_indexes = [
        row for row in index_rows if row.format_version == MAP_UNIT_INDEX_FORMAT_VERSION
    ]
    if not compatible_indexes:
        return {}
    average_idf_path = combine_average_idf(
        [(row.average_idf_path, row.unit_count) for row in compatible_indexes]
    )
    average_idf_content = combine_average_idf(
        [(row.average_idf_content, row.unit_count) for row in compatible_indexes]
    )
    path_document_count = sum(row.path_document_count or 0 for row in compatible_indexes)
    path_total_length = sum(row.path_total_length or 0 for row in compatible_indexes)
    content_document_count = sum(
        row.content_document_count or 0 for row in compatible_indexes
    )
    content_total_length = sum(row.content_total_length or 0 for row in compatible_indexes)

    unit_dicts = [
        {
            "map_unit_id": row.id,
            "path_length": row.path_token_count,
            "content_length": row.content_token_count,
        }
        for row in unit_rows
    ]
    path_stats = build_channel_bm25_stats(
        unit_rows=unit_dicts,
        map_unit_id_field="map_unit_id",
        length_field="path_length",
        channel="path",
        query_tokens=query_tokens,
        frequencies=frequencies,
        average_idf=average_idf_path,
        document_count_override=path_document_count or None,
        total_length_override=path_total_length or None,
    )
    content_stats = build_channel_bm25_stats(
        unit_rows=unit_dicts,
        map_unit_id_field="map_unit_id",
        length_field="content_length",
        channel="content",
        query_tokens=query_tokens,
        frequencies=frequencies,
        average_idf=average_idf_content,
        document_count_override=content_document_count or None,
        total_length_override=content_total_length or None,
    )
    score_units = [
        PersistedScoreUnit(
            unit_id=row.id,
            path_length=row.path_token_count,
            content_length=row.content_token_count,
            path_frequencies=frequencies.get((row.id, "path"), {}),
            content_frequencies=frequencies.get((row.id, "content"), {}),
        )
        for row in unit_rows
    ]
    corpus = PersistedScoreCorpus(
        units=score_units, path_stats=path_stats, content_stats=content_stats
    )
    scored = score_persisted_corpus_many(corpus, [query])
    unit_scores = scored.get(query, {})

    section_scores: dict[str, float] = {}
    for row in unit_rows:
        score = float(unit_scores.get(row.id, 0.0) or 0.0)
        if score > section_scores.get(row.section_id, 0.0):
            section_scores[row.section_id] = score
    return section_scores


def pool_scores_to_tree(
    *,
    children_by_parent: dict[str, list[str]],
    roots: list[str],
    leaf_scores: dict[str, float],
) -> dict[str, float]:
    """Post-order max-pool: an internal node's score is the best of its own
    leaf score (if it owns a unit) and every descendant's score."""
    pooled: dict[str, float] = {}

    def visit(section_id: str) -> float:
        if section_id in pooled:
            return pooled[section_id]
        children = children_by_parent.get(section_id) or []
        best = leaf_scores.get(section_id, 0.0)
        for child in children:
            best = max(best, visit(child))
        pooled[section_id] = best
        return best

    for root in roots:
        visit(root)
    return pooled


@dataclass
class MapNode:
    """One row of a rendered outline/node_filter map."""

    section_id: str
    parent_section_id: str | None
    render: str
    score: float = 0.0
    hidden: bool = False
    render_chars: int = field(init=False)

    def __post_init__(self) -> None:
        self.render_chars = len(self.render) + 1


def fold_map_nodes(
    nodes: list[MapNode],
    *,
    char_budget: int,
) -> tuple[list[MapNode], dict[str, int]]:
    """Hide lowest-score subtrees until the total rendering fits ``char_budget``.

    Never hides a node in isolation — only whole subtrees rooted at a node
    with no other visible ancestor already hidden — and never truncates a
    single row's text. Nodes are chosen lowest-score-first (largest subtree
    first among ties), so a positive-scoring branch is only touched once
    every zero/low-scoring branch has already been hidden and the budget
    still is not met — this is how "the highest-scoring nodes and their
    ancestor chain stay visible" holds without a separate hardcoded
    protected-count.

    Returns the still-visible nodes (original order) and, per section_id
    whose children were folded, how many descendant nodes were hidden under
    it (``"<root>"`` for a hidden top-level scope target) — the caller
    renders that count as one placeholder line under that node.
    """
    total_chars = sum(node.render_chars for node in nodes)
    if total_chars <= char_budget or not nodes:
        return nodes, {}

    by_id = {node.section_id: node for node in nodes}
    children_by_parent: dict[str, list[str]] = {}
    for node in nodes:
        if node.parent_section_id:
            children_by_parent.setdefault(node.parent_section_id, []).append(
                node.section_id
            )

    def subtree_ids(section_id: str) -> list[str]:
        ids = [section_id]
        for child_id in children_by_parent.get(section_id, []):
            ids.extend(subtree_ids(child_id))
        return ids

    def subtree_chars(section_id: str) -> int:
        return sum(
            by_id[sid].render_chars
            for sid in subtree_ids(section_id)
            if sid in by_id and not by_id[sid].hidden
        )

    hidden_counts: dict[str, int] = {}
    remaining = total_chars
    candidates = sorted(
        nodes, key=lambda node: (node.score, -subtree_chars(node.section_id), node.section_id)
    )
    for candidate in candidates:
        if remaining <= char_budget:
            break
        if candidate.hidden:
            continue
        ids = [sid for sid in subtree_ids(candidate.section_id) if sid in by_id]
        removable = [sid for sid in ids if not by_id[sid].hidden]
        if not removable:
            continue
        removed_chars = sum(by_id[sid].render_chars for sid in removable)
        for sid in removable:
            by_id[sid].hidden = True
        remaining -= removed_chars
        placeholder_key = candidate.parent_section_id or "<root>"
        hidden_counts[placeholder_key] = hidden_counts.get(placeholder_key, 0) + len(
            removable
        )

    visible = [node for node in nodes if not node.hidden]
    return visible, hidden_counts
