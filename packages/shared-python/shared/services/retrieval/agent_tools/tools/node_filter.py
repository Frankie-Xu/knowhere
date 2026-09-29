"""``corpus.node_filter`` — deterministic FOR-ALL/EXISTS/ANY/NOT predicate over sections.

Reuses the exact predicate compile/match semantics from
``scoring.node_filter_predicates`` (path/summary substring|regex, fields AND
together, terms OR together) — see that module's docstring — but walks
``document_sections`` rows for the requested scope's current revision
instead of the in-memory map-nav tree.

Unlike a plain filter, the rendered result is not just the matched rows: it
is each matched section's full ancestor chain (context) plus its entire
descendant subtree, in the same map-row shape ``corpus.outline`` uses (see
``agent_tools.snippet.format_row``) — this is a map-narrowing tool, the same
family as ``outline``, not a search hit list. Matched rows are marked
``[Hit]``. Past ``registry.MAP_TOOL_CHAR_BUDGET`` chars the rendered map is
folded the same way ``outline``'s is (see ``scoring.map_lighting``): a hit
and its ancestor chain are never folded away, but a large descendant subtree
under a hit can still have its own low-relevance branches hidden behind a
placeholder.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from shared.models.database.document import DocumentChunk, DocumentSection
from shared.services.retrieval.agent_tools.registry import (
    MAP_TOOL_CHAR_BUDGET,
    ToolContext,
    ToolResult,
    register_tool,
)
from shared.services.retrieval.agent_tools.scope import (
    SCOPE_SCHEMA,
    resolve_scope,
    scope_document_ids,
)
from shared.services.retrieval.agent_tools.snippet import format_row
from shared.services.retrieval.scoring.map_lighting import (
    MapNode,
    fold_map_nodes,
    load_leaf_unit_scores,
    pool_scores_to_tree,
)
from shared.services.retrieval.scoring.node_filter_predicates import (
    FieldPredicate,
    _compile_predicates,
    _node_matches,
    field_predicate,
)


@register_tool(
    name="corpus.node_filter",
    description=(
        "Deterministic FOR-ALL/EXISTS/ANY/NOT filter over section titles and "
        "summaries — not body text (use corpus.grep for that). Narrows a map "
        "the same way corpus.outline does: the result is each matched "
        "section's full ancestor chain plus its entire descendant subtree "
        "(marked [Hit] where matched), not a bare list of hits. Call with "
        "scope plus a predicates array; each predicate is {field: "
        "'path'|'summary', terms: [...], match: 'substring'|'regex'}. "
        "field=path matches the section path; field=summary matches the "
        "section summary. Predicates AND together across the array; terms "
        "within one predicate's 'terms' list OR together. After node_filter "
        "returns, either call corpus.finish citing section_paths you "
        "already have, or corpus.read the section_path(s) that matter. A "
        "result over 20,000 chars is folded, not cut off — see "
        "corpus.outline's description for how."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "scope": SCOPE_SCHEMA,
            "predicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["path", "summary"]},
                        "terms": {"type": "array", "items": {"type": "string"}},
                        "match": {
                            "type": "string",
                            "enum": ["substring", "regex"],
                            "default": "substring",
                        },
                    },
                    "required": ["field", "terms"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
            "chunk_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Narrow matched sections to those owning a body chunk of "
                    "one of these chunk_type values (e.g. ['page'] to filter "
                    "to page-track leaves only). Omit for no narrowing."
                ),
            },
        },
        "required": ["scope", "predicates"],
        "additionalProperties": False,
    },
)
async def node_filter(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    targets, scope_error = await resolve_scope(
        ctx.db,
        user_id=ctx.user_id,
        namespace=ctx.namespace,
        document_scope=ctx.document_scope,
        raw_scope=args.get("scope"),
    )
    if scope_error is not None:
        return ToolResult(text="", error=f"node_filter: {scope_error}")

    raw_predicates = args.get("predicates") or []
    if not raw_predicates:
        return ToolResult(text="", error="node_filter requires predicates")

    predicates: list[FieldPredicate] = []
    for raw in raw_predicates:
        try:
            predicates.append(
                field_predicate(
                    raw.get("field"),
                    raw.get("terms") or [],
                    raw.get("match", "substring"),
                )
            )
        except ValueError as exc:
            return ToolResult(text="", error=str(exc))

    compiled, failed = _compile_predicates(predicates)
    if failed:
        return ToolResult(
            text=f"failed_predicates={failed}",
            payload={"cardinality": 0, "matched_sections": [], "failed_predicates": failed},
            error="one or more predicates failed to compile",
        )

    document_ids = scope_document_ids(targets)
    revision_by_doc = {target.document_id: target.job_result_id for target in targets}

    chunk_types = {
        str(t).strip().lower() for t in (args.get("chunk_types") or []) if str(t).strip()
    }
    allowed_section_ids: set[str] | None = None
    if chunk_types:
        chunk_rows = await ctx.db.execute(
            select(DocumentChunk.section_id, DocumentChunk.chunk_type).where(
                DocumentChunk.document_id.in_(document_ids),
                DocumentChunk.job_result_id.in_(list(revision_by_doc.values())),
            )
        )
        allowed_section_ids = {
            str(section_id)
            for section_id, chunk_type in chunk_rows.all()
            if section_id and str(chunk_type or "").strip().lower() in chunk_types
        }

    matched_sections: list[dict[str, Any]] = []
    matched_doc_ids: list[str] = []
    seen_docs: set[str] = set()
    all_map_nodes: list[MapNode] = []
    all_nodes: list[dict[str, Any]] = []

    for target in targets:
        section_stmt = (
            select(DocumentSection)
            .where(DocumentSection.document_id == target.document_id)
            .where(DocumentSection.job_result_id == target.job_result_id)
            .order_by(DocumentSection.sort_order, DocumentSection.section_id)
        )
        doc_sections = list((await ctx.db.execute(section_stmt)).scalars().all())
        if target.section_path is not None:
            doc_sections = [
                s
                for s in doc_sections
                if s.section_path == target.section_path
                or s.section_path.startswith(f"{target.section_path} / ")
            ]
        by_id = {s.section_id: s for s in doc_sections}

        hit_ids: set[str] = set()
        for section in doc_sections:
            if allowed_section_ids is not None and section.section_id not in allowed_section_ids:
                continue
            values = {"path": section.section_path, "summary": section.summary or ""}
            if _node_matches(values, compiled):
                hit_ids.add(section.section_id)

        if not hit_ids:
            continue

        visible_ids: set[str] = set()
        for hit_id in hit_ids:
            node = by_id.get(hit_id)
            while node is not None:
                visible_ids.add(node.section_id)
                node = by_id.get(node.parent_section_id) if node.parent_section_id else None
            hit_path = by_id[hit_id].section_path
            for section in doc_sections:
                if section.section_path == hit_path or section.section_path.startswith(
                    f"{hit_path} / "
                ):
                    visible_ids.add(section.section_id)

        for section in matched_doc_order(doc_sections, visible_ids):
            is_hit = section.section_id in hit_ids
            if is_hit:
                matched_sections.append(
                    {
                        "document_id": target.document_id,
                        "section_id": section.section_id,
                        "section_path": section.section_path,
                        "summary": section.summary or "",
                    }
                )
                if target.document_id not in seen_docs:
                    seen_docs.add(target.document_id)
                    matched_doc_ids.append(target.document_id)
            parent_id = (
                section.parent_section_id
                if section.parent_section_id in visible_ids
                else None
            )
            all_nodes.append(
                {
                    "document_id": target.document_id,
                    "section_id": section.section_id,
                    "section_path": section.section_path,
                    "is_hit": is_hit,
                }
            )
            render = format_row(
                kind="section",
                document_id=target.document_id,
                section_path=section.section_path,
                title=section.section_title,
                summary=section.summary or "",
                is_hit=is_hit,
            )
            all_map_nodes.append(
                MapNode(section_id=section.section_id, parent_section_id=parent_id, render=render)
            )

    if not matched_sections:
        return ToolResult(
            text="hits=0",
            payload={"cardinality": 0, "matched_sections": [], "matched_document_ids": []},
            refs=[],
        )

    total_chars = sum(node.render_chars for node in all_map_nodes)
    hidden_counts: dict[str, int] = {}
    if total_chars > MAP_TOOL_CHAR_BUDGET:
        if not ctx.query.strip():
            return ToolResult(
                text="",
                error=(
                    "node_filter: result exceeds "
                    f"{MAP_TOOL_CHAR_BUDGET} chars and no query is available to "
                    "score which sections matter most — narrow scope or "
                    "predicates and try again."
                ),
            )
        section_ids = [node.section_id for node in all_map_nodes]
        leaf_scores = await load_leaf_unit_scores(
            ctx.db,
            revision_by_document=revision_by_doc,
            section_ids=section_ids,
            query=ctx.query,
        )
        hit_ids_all = {entry["section_id"] for entry in matched_sections}
        for hit_id in hit_ids_all:
            leaf_scores[hit_id] = float("inf")
        children_by_parent: dict[str, list[str]] = {}
        for node in all_map_nodes:
            if node.parent_section_id:
                children_by_parent.setdefault(node.parent_section_id, []).append(
                    node.section_id
                )
        roots = [node.section_id for node in all_map_nodes if node.parent_section_id is None]
        pooled = pool_scores_to_tree(
            children_by_parent=children_by_parent, roots=roots, leaf_scores=leaf_scores
        )
        for node in all_map_nodes:
            node.score = pooled.get(node.section_id, 0.0)
        all_map_nodes, hidden_counts = fold_map_nodes(
            all_map_nodes, char_budget=MAP_TOOL_CHAR_BUDGET
        )

    visible_map_ids = {node.section_id for node in all_map_nodes}
    ordered_lines: list[str] = []
    for node in all_map_nodes:
        ordered_lines.append(node.render)
        hidden_here = hidden_counts.get(node.section_id)
        if hidden_here:
            header_line = node.render.split("\n", 1)[0]
            leading_spaces = len(header_line) - len(header_line.lstrip(" "))
            ordered_lines.append(
                " " * (leading_spaces + 2)
                + f"(hidden {hidden_here} nodes under this section — "
                "re-scope to its section_path to expand)"
            )

    visible_matched = [
        entry for entry in matched_sections if entry["section_id"] in visible_map_ids
    ]
    header = f"hits={len(matched_sections)}"
    if hidden_counts:
        header += (
            f" (folded — {sum(hidden_counts.values())} context nodes hidden; "
            f"{len(visible_matched)}/{len(matched_sections)} hits still visible)"
        )

    return ToolResult(
        text=header + "\n" + "\n".join(ordered_lines),
        payload={
            "cardinality": len(matched_sections),
            "matched_sections": matched_sections,
            "matched_document_ids": matched_doc_ids,
        },
        refs=[
            {"document_id": entry["document_id"], "section_path": entry["section_path"]}
            for entry in visible_matched
        ],
    )


def matched_doc_order(
    doc_sections: list[DocumentSection], visible_ids: set[str]
) -> list[DocumentSection]:
    """``doc_sections`` filtered to ``visible_ids``, preserving sort order."""
    return [s for s in doc_sections if s.section_id in visible_ids]
