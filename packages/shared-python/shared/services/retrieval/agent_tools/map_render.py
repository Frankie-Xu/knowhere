"""Render an ``outline``/``node_filter`` map within ``MAP_TOOL_CHAR_BUDGET``.

A map that fits is returned whole. An oversized one is lit and folded
(``scoring.map_lighting``): sections are scored against the end user's
original query (``ToolContext.query``), the protected rows (outline: the
top-scored sections; node_filter: the predicate matches) are marked
``[Hit]`` and kept with their ancestor chain, and low-scoring subtrees are
replaced by a placeholder row. When that is impossible — no query, no
usable map-unit index, nothing matches, or the protected rows alone
overflow — the call fails with a narrow-the-scope error. The text is never
cut mid-row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shared.services.retrieval.agent_tools.registry import MAP_TOOL_CHAR_BUDGET, ToolContext
from shared.services.retrieval.agent_tools.snippet import format_row
from shared.services.retrieval.scoring.map_lighting import (
    MapNode,
    fold_map_nodes,
    load_leaf_unit_scores,
    pool_scores_to_tree,
    render_map_lines,
    top_hit_ids,
)


@dataclass
class MapRender:
    text: str = ""
    visible_ids: set[str] = field(default_factory=set)
    error: str | None = None


def _nodes(
    order: list[tuple[str, str | None]],
    rows_by_id: dict[str, dict[str, Any]],
    scores: dict[str, float] | None = None,
) -> list[MapNode]:
    return [
        MapNode(
            section_id=section_id,
            parent_section_id=parent_id,
            render=format_row(rows_by_id[section_id]),
            depth=rows_by_id[section_id]["depth"],
            score=(scores or {}).get(section_id, 0.0),
        )
        for section_id, parent_id in order
    ]


def _narrow_error(tool: str, reason: str) -> str:
    return (
        f"{tool}: {reason} Narrow the call: add a section_path to scope, "
        "pass a smaller depth, or split the documents across calls. To find "
        "a specific fact instead of mapping, use corpus.grep or corpus.recall."
    )


async def render_map(
    ctx: ToolContext,
    *,
    tool: str,
    order: list[tuple[str, str | None]],
    rows_by_id: dict[str, dict[str, Any]],
    revision_by_document: dict[str, str],
    header: Callable[[int, int], str],
    protected_ids: set[str] | None = None,
) -> MapRender:
    """``order`` lists ``(section_id, in-map parent id)`` in display order.

    ``header(visible_count, hidden_count)`` renders the first line.
    ``protected_ids=None`` means "protect the top-scored sections" (outline);
    a set means "protect exactly these" (node_filter's matches, already
    marked ``is_hit`` in ``rows_by_id``).
    """
    nodes = _nodes(order, rows_by_id)
    lines = render_map_lines(nodes)
    text = header(len(nodes), 0) + "\n" + "\n".join(lines)
    if len(text) <= MAP_TOOL_CHAR_BUDGET:
        return MapRender(text=text, visible_ids={node.section_id for node in nodes})

    over = f"the map is {len(text)} chars, over the {MAP_TOOL_CHAR_BUDGET}-char limit, and"
    if not ctx.query.strip():
        return MapRender(
            error=_narrow_error(tool, f"{over} no user query is available to rank sections.")
        )
    leaf_scores = await load_leaf_unit_scores(
        ctx.db,
        revision_by_document=revision_by_document,
        section_ids=[section_id for section_id, _ in order],
        query=ctx.query,
    )
    if leaf_scores is None:
        return MapRender(
            error=_narrow_error(
                tool,
                f"{over} these documents have no usable relevance index to "
                "decide which sections to fold.",
            )
        )

    if protected_ids is None:
        protected_ids = set(top_hit_ids(leaf_scores))
        if not protected_ids:
            return MapRender(
                error=_narrow_error(
                    tool, f"{over} no section in this scope matches the user query."
                )
            )
        for section_id in protected_ids:
            rows_by_id[section_id]["is_hit"] = True

    children_by_parent: dict[str, list[str]] = {}
    for section_id, parent_id in order:
        if parent_id is not None:
            children_by_parent.setdefault(parent_id, []).append(section_id)
    pooled = pool_scores_to_tree(
        children_by_parent=children_by_parent,
        roots=[section_id for section_id, parent_id in order if parent_id is None],
        leaf_scores=leaf_scores,
    )
    nodes = _nodes(order, rows_by_id, pooled)
    widest_header = header(len(nodes), len(nodes))
    folded = fold_map_nodes(
        nodes,
        char_budget=MAP_TOOL_CHAR_BUDGET - len(widest_header) - 1,
        protected_ids=protected_ids,
    )
    if folded.overflow:
        return MapRender(
            error=_narrow_error(
                tool,
                f"{over} the matched sections and their parents alone do not "
                "fit even after folding everything else.",
            )
        )
    return MapRender(
        text=header(len(folded.visible_ids), folded.hidden_count)
        + "\n"
        + "\n".join(folded.lines),
        visible_ids=folded.visible_ids,
    )
