"""``corpus.recall`` — fuzzy ranked candidate search.

One live scoring path: ``search.map_unit_discovery.map_unit_discovery``
(persisted map-unit BM25 over path + content, already RRF-fused there).
Exact-string lookup is ``corpus.grep``, not a second channel here.

A ``scope`` section-subtree restriction is applied *after* that call, over
a pool of up to ``budget.max_items`` candidates, not as a SQL predicate
inside ``map_unit_discovery`` itself — that shared function backs classic
retrieval as well, and adding a new SQL clause there was judged a
materially bigger, riskier change than this tool warrants. In the rare
case where a scoped subtree's true best matches fall outside that pool,
this under-returns rather than searching outside the scope — disclosed
here, not silent.

Rows render through the shared ``agent_tools.snippet.format_row`` — the
same row shape ``corpus.outline``/``corpus.node_filter``/``corpus.grep``/
``corpus.assets`` use.
"""

from __future__ import annotations

from typing import Any

from shared.services.retrieval.agent_tools.explore_mount import mount_explore_hits
from shared.services.retrieval.agent_tools.registry import (
    ToolContext,
    ToolResult,
    capped_limit,
    register_tool,
)
from shared.services.retrieval.agent_tools.scope import (
    SCOPE_SCHEMA,
    ScopeTarget,
    resolve_scope,
    scope_document_ids,
)
from shared.services.retrieval.agent_tools.snippet import build_snippet, format_row
from shared.services.retrieval.hydration.row_utils import normalize_chunk_type
from shared.services.retrieval.settings import ASSET_CHUNK_TYPES
from shared.services.retrieval.search.map_unit_discovery import map_unit_discovery

_DEFAULT_LIMIT = 10


def _in_scope(scope: list[ScopeTarget], *, document_id: str, section_path: str) -> bool:
    if not scope:
        return True
    for target in scope:
        if target.document_id != document_id:
            continue
        if target.section_path is None:
            return True
        if section_path == target.section_path or section_path.startswith(
            f"{target.section_path} / "
        ):
            return True
    return False


def _identifier_snippet(row: dict[str, Any]) -> str:
    """Body/image fall back to content; tables never use the stored path."""
    snippet = str(row.get("snippet") or "").strip()
    if snippet:
        return snippet
    if normalize_chunk_type(row.get("chunk_type")) == "table":
        metadata = row.get("chunk_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        summary = str(metadata.get("summary") or "").strip()
        return build_snippet(summary) if summary else ""
    return build_snippet(str(row.get("content") or ""))


@register_tool(
    name="corpus.recall",
    description=(
        "Fuzzy ranked candidate search for a question when you don't know "
        "where the answer lives. Scores path and content with BM25. "
        "Returns candidates with chunk_type, document_id, path and snippet. "
        "Table and image hits include rendered content; body hits include "
        "any connected table or image. Exact string / identifier lookup is "
        "corpus.grep, not this tool. Read the candidate's document_id + "
        "section_path (or chunk_id, for image/table hits) with corpus.read "
        "next, not the filename."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "scope": SCOPE_SCHEMA,
            "chunk_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": _DEFAULT_LIMIT},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
async def recall(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult(text="", error="recall requires query")
    requested_limit = int(args.get("limit") or _DEFAULT_LIMIT)
    limit = capped_limit(requested_limit, ctx.budget)
    chunk_types = {
        str(t).strip().lower() for t in (args.get("chunk_types") or []) if str(t).strip()
    } or None

    scope: list[ScopeTarget] = []
    if args.get("scope") is not None:
        scope, scope_error = await resolve_scope(
            ctx.db,
            user_id=ctx.user_id,
            namespace=ctx.namespace,
            document_scope=ctx.document_scope,
            raw_scope=args.get("scope"),
        )
        if scope_error is not None:
            return ToolResult(text="", error=f"recall: {scope_error}")

    document_scope = (
        ctx.document_scope.narrow(scope_document_ids(scope)) if scope else ctx.document_scope
    )
    scoped_subtree = any(target.section_path is not None for target in scope)

    # See module docstring: subtree scoping is a post-filter over a
    # bounded pool here, not a SQL predicate inside map_unit_discovery.
    discovery_top_k = ctx.budget.max_items if scoped_subtree else limit
    discovery = await map_unit_discovery(
        ctx.db,
        user_id=ctx.user_id,
        namespace=ctx.namespace,
        query=query,
        top_k=discovery_top_k,
        exclude_document_ids=[],
        document_scope=document_scope,
        exclude_sections=[],
        chunk_types=chunk_types,
    )
    rows = list(discovery.payload.get("fused_rows") or [])
    if scoped_subtree:
        rows = [
            row
            for row in rows
            if _in_scope(
                scope,
                document_id=str(row.get("document_id") or ""),
                section_path=str(row.get("section_path") or ""),
            )
        ][:limit]

    media: list[dict[str, str]] = []
    if rows:
        rows, media = await mount_explore_hits(
            ctx, rows, char_budget=ctx.budget.max_chars
        )

    lines = [f"candidates={len(rows)}"]
    if len(rows) < 2:
        # No tool name named here on purpose — this fires on *every* weak
        # recall regardless of what a better next step happens to be for
        # this corpus/query, so it nudges the agent to change approach
        # without prescribing which other tool to reach for (that's already
        # covered generically in CORPUS_SCHEMA.md §6's tool-selection table).
        lines.append(
            "note: few or no candidates for this phrasing — rephrasing the "
            "query and calling recall again rarely surfaces more; a "
            "different exploration approach is more likely to help than "
            "repeating recall with synonyms."
        )
    if requested_limit > limit:
        lines.append(f"note: capped to budget.max_items={ctx.budget.max_items}")
    for row in rows:
        snippet = _identifier_snippet(row)
        chunk_type = str(row.get("chunk_type") or "").strip()
        lines.append(
            format_row(
                kind=chunk_type or "text",
                document_id=row.get("document_id"),
                section_path=row.get("section_path"),
                title=row.get("source_file_name"),
                chunk_id=(
                    row.get("chunk_id") if chunk_type in ASSET_CHUNK_TYPES else None
                ),
                snippet=snippet,
                score=row.get("score"),
            )
        )
        rendered = str(row.get("rendered") or "").strip()
        if rendered:
            lines.append(rendered)

    return ToolResult(
        text="\n".join(lines),
        payload={"candidates": rows},
        refs=[
            {"document_id": row.get("document_id"), "chunk_id": row.get("chunk_id")}
            for row in rows
        ],
        media=media,
    )
