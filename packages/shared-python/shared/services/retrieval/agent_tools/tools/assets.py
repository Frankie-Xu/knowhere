"""``corpus.assets`` — forward asset search and reverse asset -> hosts lookup.

Image/table chunks are parked under their document's synthetic ``Root``
section in the DB (§3 of ``CORPUS_SCHEMA.md``); the real association to a
body section lives in ``chunk_metadata.connect_to`` on the *body* chunk, not
on the asset. There is no stored asset -> body back-link, so the reverse
lookup (``host_of``) scans the candidate documents' text/page chunks in
Python and checks ``connect_to`` for the requested target ids. Forward
search rows show the asset's *hosting* section_path (via that same reverse
scan), not the DB's literal ``Root`` — this is also how a ``scope`` section
subtree narrows forward search: an asset is in scope when some in-scope body
chunk connects to it, not by the asset's own (always-Root) stored path.

``chunk_metadata`` is a plain ``JSON`` column (not ``JSONB``), so a
containment query (``@>``) is not available here — that operator is
JSONB-only in PostgreSQL.

Forward search on its own is an unfiltered listing with no way to decide
which assets matter for the current question — pair it with a prior
``corpus.recall``/``corpus.grep`` hit (scope to that hit's section, or use
its ``host_of`` reverse lookup) rather than browsing every asset in a
document.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from shared.models.database.document import Document, DocumentChunk, DocumentSection
from shared.services.retrieval.agent_tools.registry import (
    ToolContext,
    ToolResult,
    register_tool,
)
from shared.services.retrieval.agent_tools.scope import (
    SCOPE_SCHEMA,
    ScopeTarget,
    resolve_scope,
    scope_document_ids,
)
from shared.services.retrieval.agent_tools.snippet import format_row
from shared.services.retrieval.hydration.row_utils import iter_connected_target_ids
from shared.services.retrieval.settings import ASSET_CHUNK_TYPES

_BODY_CHUNK_TYPES = ("text", "page")


def _host_in_scope(
    scope: list[ScopeTarget], *, document_id: str, section_path: str | None
) -> bool:
    for target in scope:
        if target.document_id != document_id:
            continue
        if target.section_path is None:
            return True
        path = section_path or ""
        if path == target.section_path or path.startswith(f"{target.section_path} / "):
            return True
    return False


@register_tool(
    name="corpus.assets",
    description=(
        "Forward search for image/table chunks by type/query, or reverse "
        "lookup: given asset chunk_ids (host_of), find which body "
        "section(s) embed or reference them via connect_to. A forward "
        "search alone only lists candidate assets — it does not tell you "
        "which ones matter for the current question. Pair it with a prior "
        "corpus.recall/corpus.grep hit: scope this call to that hit's "
        "section, or reverse-resolve the hit's own connect_to targets via "
        "host_of instead of browsing every asset in a document."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "scope": SCOPE_SCHEMA,
            "type": {
                "type": "string",
                "enum": ["image", "table", "any"],
                "default": "any",
            },
            "query": {
                "type": "string",
                "description": "Substring match against summary/keywords (forward search only).",
            },
            "host_of": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Asset chunk_ids to reverse-resolve to hosting sections.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)
async def assets(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
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
            return ToolResult(text="", error=f"assets: {scope_error}")
    host_of = [str(c).strip() for c in (args.get("host_of") or []) if str(c).strip()]

    document_ids = scope_document_ids(scope)
    scope_filters: list[Any] = [
        ctx.document_scope.predicate(Document.document_id),
        Document.user_id == ctx.user_id,
        Document.namespace == ctx.namespace,
        Document.status == "active",
        Document.current_job_result_id == DocumentChunk.job_result_id,
    ]
    if document_ids:
        scope_filters.append(Document.document_id.in_(document_ids))

    if host_of:
        return await _reverse_lookup(ctx, scope_filters=scope_filters, target_ids=host_of)
    return await _forward_search(
        ctx,
        scope_filters=scope_filters,
        scope=scope,
        asset_type=str(args.get("type") or "any").strip().lower(),
        query=str(args.get("query") or "").strip().lower(),
    )


async def _forward_search(
    ctx: ToolContext,
    *,
    scope_filters: list[Any],
    scope: list[ScopeTarget],
    asset_type: str,
    query: str,
) -> ToolResult:
    types = {asset_type} if asset_type in ASSET_CHUNK_TYPES else set(ASSET_CHUNK_TYPES)

    allowed_asset_ids: set[str] | None = None
    if any(target.section_path is not None for target in scope):
        body_stmt = (
            select(DocumentChunk, DocumentSection.section_path)
            .select_from(DocumentChunk)
            .join(Document, Document.document_id == DocumentChunk.document_id)
            .outerjoin(
                DocumentSection, DocumentSection.section_id == DocumentChunk.section_id
            )
            .where(*scope_filters)
            .where(DocumentChunk.chunk_type.in_(_BODY_CHUNK_TYPES))
        )
        body_rows = (await ctx.db.execute(body_stmt)).all()
        allowed_asset_ids = set()
        for chunk, section_path in body_rows:
            if not _host_in_scope(
                scope, document_id=chunk.document_id, section_path=section_path
            ):
                continue
            row = {"chunk_metadata": chunk.chunk_metadata}
            allowed_asset_ids.update(iter_connected_target_ids(row))
        if not allowed_asset_ids:
            return ToolResult(text="assets=0", payload={"assets": []}, refs=[])

    stmt = (
        select(DocumentChunk, DocumentSection.section_path, Document.source_file_name)
        .select_from(DocumentChunk)
        .join(Document, Document.document_id == DocumentChunk.document_id)
        .outerjoin(DocumentSection, DocumentSection.section_id == DocumentChunk.section_id)
        .where(*scope_filters)
        .where(DocumentChunk.chunk_type.in_(sorted(types)))
        .order_by(DocumentChunk.document_id, DocumentChunk.sort_order)
    )
    if allowed_asset_ids is not None:
        stmt = stmt.where(DocumentChunk.chunk_id.in_(allowed_asset_ids))
    rows = (await ctx.db.execute(stmt)).all()

    # Reverse-scan once for every candidate document to find each asset's
    # hosting section_path (the DB's own section_path for an asset chunk is
    # always Root — see module docstring).
    document_ids = sorted({chunk.document_id for chunk, _section_path, _name in rows})
    host_by_asset_id: dict[str, str] = {}
    if document_ids:
        host_stmt = (
            select(DocumentChunk, DocumentSection.section_path)
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id.in_(document_ids))
            .where(DocumentChunk.chunk_type.in_(_BODY_CHUNK_TYPES))
            .outerjoin(
                DocumentSection, DocumentSection.section_id == DocumentChunk.section_id
            )
        )
        for chunk, section_path in (await ctx.db.execute(host_stmt)).all():
            row = {"chunk_metadata": chunk.chunk_metadata}
            for target_id in iter_connected_target_ids(row):
                host_by_asset_id.setdefault(target_id, section_path or "Root")

    results: list[dict[str, Any]] = []
    for chunk, _section_path, source_file_name in rows:
        metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
        summary = str(metadata.get("summary") or "").strip()
        keywords = metadata.get("keywords") or []
        if query:
            haystack = " ".join(
                [summary.lower(), " ".join(str(k).lower() for k in keywords)]
            )
            if query not in haystack:
                continue
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_file_name": source_file_name,
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "summary": summary,
                "keywords": keywords,
                "section_path": host_by_asset_id.get(chunk.chunk_id, "Root"),
            }
        )
        if len(results) >= ctx.budget.max_items:
            break

    lines = [f"assets={len(results)}"]
    for r in results:
        lines.append(
            format_row(
                kind=r["chunk_type"],
                document_id=r["document_id"],
                section_path=r["section_path"],
                title=r["source_file_name"],
                chunk_id=r["chunk_id"],
                summary=r["summary"],
            )
        )

    return ToolResult(
        text="\n".join(lines),
        payload={"assets": results},
        refs=[{"document_id": r["document_id"], "chunk_id": r["chunk_id"]} for r in results],
    )


async def _reverse_lookup(
    ctx: ToolContext,
    *,
    scope_filters: list[Any],
    target_ids: list[str],
) -> ToolResult:
    target_set = set(target_ids)
    stmt = (
        select(DocumentChunk, DocumentSection.section_path, Document.source_file_name)
        .select_from(DocumentChunk)
        .join(Document, Document.document_id == DocumentChunk.document_id)
        .outerjoin(DocumentSection, DocumentSection.section_id == DocumentChunk.section_id)
        .where(*scope_filters)
        .where(DocumentChunk.chunk_type.in_(_BODY_CHUNK_TYPES))
    )
    rows = (await ctx.db.execute(stmt)).all()

    hosts_by_target: dict[str, list[dict[str, Any]]] = {tid: [] for tid in target_set}
    for chunk, section_path, source_file_name in rows:
        row = {"chunk_metadata": chunk.chunk_metadata}
        for target_id in iter_connected_target_ids(row):
            if target_id in target_set:
                hosts_by_target[target_id].append(
                    {
                        "document_id": chunk.document_id,
                        "source_file_name": source_file_name,
                        "section_path": section_path,
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                    }
                )

    lines = []
    for target_id, hosts in hosts_by_target.items():
        if not hosts:
            lines.append(f"- {target_id}: no host found (unresolved Root asset)")
            continue
        for host in hosts:
            lines.append(
                format_row(
                    kind=host["chunk_type"],
                    document_id=host["document_id"],
                    section_path=host["section_path"],
                    title=host["source_file_name"],
                    chunk_id=target_id,
                )
            )

    return ToolResult(
        text="\n".join(lines) if lines else "no hosts found",
        payload={"hosts_by_target": hosts_by_target},
        refs=[
            {"document_id": host["document_id"], "section_path": host["section_path"]}
            for hosts in hosts_by_target.values()
            for host in hosts
        ],
    )
