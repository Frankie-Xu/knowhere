"""``corpus.outline`` — titles + summaries, no body text.

Reads ``document_sections`` (+ a ``document_chunks`` count aggregate) for one
or more scope targets' current revision, each optionally narrowed to a
section subtree. This queries the live tables directly rather than the
compressed ``RetrievalNamespaceMapSnapshot``/serving-manifest blob: that
snapshot is namespace-wide and decoding it to read one document's subtree
would cost more than this document-scoped, index-backed query.

A scope target's own row is always included, so calling this on a section
with no descendants (a leaf) would return exactly that one row with no
outline value — rejected up front instead, pointing at ``corpus.read``.

The rendered map (one row per section, same shape ``corpus.node_filter``
uses — see ``agent_tools.snippet.format_row``) is never truncated mid-text.
Past ``registry.MAP_TOOL_CHAR_BUDGET`` chars it is folded instead: low-
relevance subtrees (scored against the caller's original query — see
``ToolContext.query``) are hidden behind a placeholder, keeping every
positive-scoring branch and its ancestor chain visible. See
``scoring.map_lighting``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from shared.models.database.document import Document, DocumentChunk, DocumentSection
from shared.services.retrieval.agent_tools.registry import (
    MAP_TOOL_CHAR_BUDGET,
    ToolContext,
    ToolResult,
    register_tool,
)
from shared.services.retrieval.agent_tools.scope import (
    SCOPE_SCHEMA,
    resolve_scope,
)
from shared.services.retrieval.agent_tools.snippet import format_row
from shared.services.retrieval.scoring.map_lighting import (
    MapNode,
    fold_map_nodes,
    load_leaf_unit_scores,
    pool_scores_to_tree,
)


@register_tool(
    name="corpus.outline",
    description=(
        "Map what a document (or one of its sections) covers: every "
        "section_path beneath a scope target, any depth, with title + "
        "summary + chunk_count, indented by level — no body text. Use this "
        "for 'what does this document/section cover' questions, not for "
        "finding an answer to a specific fact (use corpus.recall/corpus.grep "
        "for that). Accepts several scope targets in one call. Each target "
        "must be a non-leaf section (or the whole document, via a scope "
        "item with no section_path) — calling this on a section with no "
        "children is rejected; read it directly with corpus.read instead. "
        "After outline returns, either call corpus.finish citing the "
        "section_paths you already have, or corpus.read the section_path(s) "
        "that matter. A result over 20,000 chars is folded, not cut off: "
        "low-relevance subtrees are hidden behind a 'hidden N nodes' "
        "placeholder — re-scope to that section_path to expand it."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "scope": SCOPE_SCHEMA,
            "depth": {
                "type": "integer",
                "description": (
                    "Max levels below each scope target to include. Omit "
                    "for unlimited depth."
                ),
            },
        },
        "required": ["scope"],
        "additionalProperties": False,
    },
)
async def outline(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    targets, scope_error = await resolve_scope(
        ctx.db,
        user_id=ctx.user_id,
        namespace=ctx.namespace,
        document_scope=ctx.document_scope,
        raw_scope=args.get("scope"),
    )
    if scope_error is not None:
        return ToolResult(text="", error=f"outline: {scope_error}")

    depth_raw = args.get("depth")
    depth = int(depth_raw) if depth_raw is not None else None
    if depth is not None and depth < 0:
        return ToolResult(text="", error="outline: depth must be >= 0")

    document_ids = sorted({target.document_id for target in targets})
    documents = (
        (
            await ctx.db.execute(
                select(Document).where(Document.document_id.in_(document_ids))
            )
        )
        .scalars()
        .all()
    )
    source_file_name_by_doc = {d.document_id: d.source_file_name or "" for d in documents}

    all_nodes: list[dict[str, Any]] = []
    map_nodes: list[MapNode] = []
    root_section_ids: list[str] = []
    section_ids_by_revision: dict[tuple[str, str], list[str]] = {}

    for target in targets:
        section_stmt = (
            select(DocumentSection)
            .where(DocumentSection.document_id == target.document_id)
            .where(DocumentSection.job_result_id == target.job_result_id)
            .order_by(DocumentSection.sort_order, DocumentSection.section_id)
        )
        doc_sections = list((await ctx.db.execute(section_stmt)).scalars().all())

        if target.section_path is None:
            base_level = 0
            scoped = doc_sections
            root_id = next(
                (s.section_id for s in doc_sections if s.section_level == 0), None
            )
        else:
            root_section = next(
                (s for s in doc_sections if s.section_path == target.section_path), None
            )
            if root_section is None:
                return ToolResult(
                    text="",
                    error=(
                        f"outline: unknown section_path for {target.document_id}: "
                        f"{target.section_path}"
                    ),
                )
            has_children = any(
                s.section_path.startswith(f"{target.section_path} / ")
                for s in doc_sections
            )
            if not has_children:
                return ToolResult(
                    text="",
                    error=(
                        f"outline: {target.section_path} in {target.document_id} is a "
                        "leaf section (no child sections) — call corpus.read on it "
                        "directly instead of outline."
                    ),
                )
            base_level = root_section.section_level
            scoped = [
                s
                for s in doc_sections
                if s.section_path == target.section_path
                or s.section_path.startswith(f"{target.section_path} / ")
            ]
            root_id = root_section.section_id

        if depth is not None:
            scoped = [s for s in scoped if (s.section_level - base_level) <= depth]
        if root_id is not None:
            root_section_ids.append(root_id)

        section_ids = [s.section_id for s in scoped]
        section_ids_by_revision[(target.document_id, target.job_result_id)] = section_ids
        scoped_ids = set(section_ids)

        chunk_counts: dict[str, int] = {}
        if section_ids:
            count_rows = await ctx.db.execute(
                select(DocumentChunk.section_id, func.count(DocumentChunk.id))
                .where(DocumentChunk.document_id == target.document_id)
                .where(DocumentChunk.job_result_id == target.job_result_id)
                .where(DocumentChunk.section_id.in_(section_ids))
                .group_by(DocumentChunk.section_id)
            )
            chunk_counts = {str(sid): int(count) for sid, count in count_rows.all()}

        for section in scoped:
            relative_depth = section.section_level - base_level
            node = {
                "document_id": target.document_id,
                "section_id": section.section_id,
                "section_path": section.section_path,
                "section_title": section.section_title,
                "section_level": section.section_level,
                "relative_depth": relative_depth,
                "summary": section.summary or "",
                "chunk_count": chunk_counts.get(section.section_id, 0),
            }
            all_nodes.append(node)
            parent_id = (
                section.parent_section_id
                if section.parent_section_id in scoped_ids
                else None
            )
            render = format_row(
                kind="section",
                document_id=target.document_id,
                section_path=section.section_path,
                title=section.section_title,
                chunk_count=node["chunk_count"],
                summary=node["summary"],
                depth=relative_depth,
            )
            map_nodes.append(
                MapNode(section_id=section.section_id, parent_section_id=parent_id, render=render)
            )

    if not all_nodes:
        return ToolResult(text="sections=0", payload={"sections": []}, refs=[])

    total_chars = sum(node.render_chars for node in map_nodes)
    hidden_counts: dict[str, int] = {}
    if total_chars > MAP_TOOL_CHAR_BUDGET:
        if not ctx.query.strip():
            return ToolResult(
                text="",
                error=(
                    "outline: result exceeds "
                    f"{MAP_TOOL_CHAR_BUDGET} chars and no query is available to "
                    "score which sections matter most — narrow scope (add a "
                    "section_path or a smaller depth) and try again."
                ),
            )
        revision_by_document = {
            target.document_id: target.job_result_id for target in targets
        }
        section_ids = [node.section_id for node in map_nodes]
        leaf_scores = await load_leaf_unit_scores(
            ctx.db,
            revision_by_document=revision_by_document,
            section_ids=section_ids,
            query=ctx.query,
        )
        children_by_parent: dict[str, list[str]] = {}
        for node in map_nodes:
            if node.parent_section_id:
                children_by_parent.setdefault(node.parent_section_id, []).append(
                    node.section_id
                )
        pooled = pool_scores_to_tree(
            children_by_parent=children_by_parent,
            roots=root_section_ids,
            leaf_scores=leaf_scores,
        )
        for node in map_nodes:
            node.score = pooled.get(node.section_id, 0.0)
        map_nodes, hidden_counts = fold_map_nodes(
            map_nodes, char_budget=MAP_TOOL_CHAR_BUDGET
        )

    visible_ids = {node.section_id for node in map_nodes}
    ordered_lines: list[str] = []
    for node in map_nodes:
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
    if hidden_counts.get("<root>"):
        ordered_lines.append(
            f"(hidden {hidden_counts['<root>']} nodes at the top level — "
            "re-scope to expand)"
        )

    visible_nodes = [node for node in all_nodes if node["section_id"] in visible_ids]
    document_names = ", ".join(
        f"{source_file_name_by_doc.get(doc_id, doc_id)} ({doc_id})"
        for doc_id in document_ids
    )
    header = f"documents={document_names} sections={len(visible_nodes)}"
    if hidden_counts:
        header += f" (folded — {sum(hidden_counts.values())} nodes hidden)"

    return ToolResult(
        text=header + "\n" + "\n".join(ordered_lines),
        payload={"sections": visible_nodes},
        refs=[
            {"document_id": node["document_id"], "section_path": node["section_path"]}
            for node in visible_nodes
        ],
    )
