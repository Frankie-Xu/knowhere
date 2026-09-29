"""outline / node_filter / scope behavior from the tool-contract plan."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("TMP_PATH", "/tmp/knowhere-test")
os.environ.setdefault("S3_BUCKET_NAME", "test-uploads")
os.environ.setdefault("S3_ACCESS_KEY_ID", "test")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("S3_TEMP_PATH", "/tmp")

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from shared.models.database.document import Document, DocumentChunk, DocumentSection
from shared.models.database.job_result import JobResult
from shared.services.retrieval.agent_tools.registry import MAP_TOOL_CHAR_BUDGET, ToolContext
from shared.services.retrieval.agent_tools.tools.assets import assets
from shared.services.retrieval.agent_tools.tools.grep import grep
from shared.services.retrieval.agent_tools.tools.node_filter import node_filter
from shared.services.retrieval.agent_tools.tools.outline import outline
from shared.services.retrieval.agent_tools.tools.recall import recall
from shared.services.retrieval.search.map_unit_discovery import DiscoveryResult
from shared.services.retrieval.scoring.map_lighting import MapNode, fold_map_nodes

USER_ID = "user_map"
NAMESPACE = "default"
DOC_A = "doc_a"
DOC_B = "doc_b"
REV_A = "jr_a"
REV_B = "jr_b"
FILE_A = "guide.pdf"
FILE_B = "notes.pdf"
PATH_ROOT = f"{FILE_A} / Root"
PATH_OVERVIEW = f"{FILE_A} / 1 Overview"
PATH_FINDINGS = f"{FILE_A} / 1 Overview / 1.1 Findings"
PATH_DETAIL = f"{FILE_A} / 1 Overview / 1.1 Findings / Detail"
PATH_TREATMENT = f"{FILE_A} / 2 Treatment"
PATH_INTRO = f"{FILE_B} / Intro"


class _AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, statement):  # noqa: ANN001
        return self._session.execute(statement)


@asynccontextmanager
async def _unused_db_factory():
    raise AssertionError("map-scope tests should not open a second session")
    yield  # pragma: no cover


def _seed(session: Session) -> None:
    now = datetime(2026, 1, 1)
    session.add_all(
        [
            JobResult(id=REV_A, job_id="job_a", delivery_mode="inline", created_at=now, updated_at=now),
            JobResult(id=REV_B, job_id="job_b", delivery_mode="inline", created_at=now, updated_at=now),
            Document(
                document_id=DOC_A,
                user_id=USER_ID,
                namespace=NAMESPACE,
                status="active",
                current_job_result_id=REV_A,
                source_file_name=FILE_A,
                parse_track="chunk",
                created_at=now,
                updated_at=now,
            ),
            Document(
                document_id=DOC_B,
                user_id=USER_ID,
                namespace=NAMESPACE,
                status="active",
                current_job_result_id=REV_B,
                source_file_name=FILE_B,
                parse_track="chunk",
                created_at=now,
                updated_at=now,
            ),
        ]
    )

    def section(
        section_id: str,
        document_id: str,
        job_result_id: str,
        path: str,
        title: str,
        level: int,
        sort_order: int,
        parent: str | None,
        summary: str = "",
    ) -> DocumentSection:
        return DocumentSection(
            section_id=section_id,
            user_id=USER_ID,
            namespace=NAMESPACE,
            document_id=document_id,
            job_result_id=job_result_id,
            parent_section_id=parent,
            section_path=path,
            section_title=title,
            section_level=level,
            summary=summary,
            sort_order=sort_order,
            created_at=now,
        )

    def chunk(
        row_id: str,
        chunk_id: str,
        document_id: str,
        job_result_id: str,
        section_id: str,
        content: str,
        *,
        chunk_type: str = "text",
        connect_to: list[dict[str, str]] | None = None,
        file_path: str | None = None,
    ) -> DocumentChunk:
        return DocumentChunk(
            id=row_id,
            chunk_id=chunk_id,
            user_id=USER_ID,
            namespace=NAMESPACE,
            document_id=document_id,
            job_result_id=job_result_id,
            section_id=section_id,
            chunk_type=chunk_type,
            content=content,
            term_search_text=content,
            file_path=file_path,
            source_chunk_path=file_path or chunk_id,
            chunk_metadata={"connect_to": connect_to or [], "summary": content},
            sort_order=0,
            created_at=now,
        )

    session.add_all(
        [
            section("sec_root", DOC_A, REV_A, PATH_ROOT, "Root", 0, 0, None),
            section("sec_overview", DOC_A, REV_A, PATH_OVERVIEW, "1 Overview", 1, 1, "sec_root"),
            section(
                "sec_findings",
                DOC_A,
                REV_A,
                PATH_FINDINGS,
                "1.1 Findings",
                2,
                2,
                "sec_overview",
                "hit-marker-findings",
            ),
            section("sec_detail", DOC_A, REV_A, PATH_DETAIL, "Detail", 3, 3, "sec_findings"),
            section("sec_treatment", DOC_A, REV_A, PATH_TREATMENT, "2 Treatment", 1, 4, "sec_root"),
            section("sec_root_b", DOC_B, REV_B, f"{FILE_B} / Root", "Root", 0, 0, None),
            section("sec_intro", DOC_B, REV_B, PATH_INTRO, "Intro", 1, 1, "sec_root_b"),
            chunk("row_findings", "chunk_findings", DOC_A, REV_A, "sec_findings", "findings body"),
            chunk("row_detail", "chunk_detail", DOC_A, REV_A, "sec_detail", "detail body"),
            chunk("row_treatment", "chunk_treatment", DOC_A, REV_A, "sec_treatment", "treatment body"),
            chunk("row_intro", "chunk_intro", DOC_B, REV_B, "sec_intro", "intro body"),
            chunk(
                "row_host_overview",
                "chunk_host_overview",
                DOC_A,
                REV_A,
                "sec_overview",
                "overview hosts table",
                connect_to=[{"target": "chunk_table_overview", "ref": "[tables/a.html]"}],
            ),
            chunk(
                "row_host_treatment",
                "chunk_host_treatment",
                DOC_A,
                REV_A,
                "sec_treatment",
                "treatment hosts table",
                connect_to=[{"target": "chunk_table_treatment", "ref": "[tables/b.html]"}],
            ),
            chunk(
                "row_table_overview",
                "chunk_table_overview",
                DOC_A,
                REV_A,
                "sec_root",
                "overview table",
                chunk_type="table",
                file_path="tables/a.html",
            ),
            chunk(
                "row_table_treatment",
                "chunk_table_treatment",
                DOC_A,
                REV_A,
                "sec_root",
                "treatment table",
                chunk_type="table",
                file_path="tables/b.html",
            ),
        ]
    )
    session.commit()


@pytest.fixture
def map_ctx() -> ToolContext:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _disable_fk(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        dbapi_connection.execute("PRAGMA foreign_keys=OFF")

    JobResult.__table__.create(engine)
    Document.__table__.create(engine)
    DocumentSection.__table__.create(engine)
    DocumentChunk.__table__.create(engine)
    session = Session(engine)
    _seed(session)
    return ToolContext(
        db=_AsyncSessionAdapter(session),  # type: ignore[arg-type]
        user_id=USER_ID,
        namespace=NAMESPACE,
        db_factory=_unused_db_factory,
        query="findings",
    )


@pytest.mark.asyncio
async def test_outline_rejects_a_leaf_section(map_ctx: ToolContext) -> None:
    result = await outline(
        map_ctx, {"scope": [{"document_id": DOC_A, "section_path": PATH_TREATMENT}]}
    )
    assert result.error is not None
    assert "leaf section" in result.error
    assert "corpus.read" in result.error


@pytest.mark.asyncio
async def test_unknown_scope_path_fails_the_call(map_ctx: ToolContext) -> None:
    result = await outline(
        map_ctx, {"scope": [{"document_id": DOC_A, "section_path": "no such path"}]}
    )
    assert result.error is not None
    assert "unknown section_path" in result.error
    assert "no such path" in result.error


@pytest.mark.asyncio
async def test_outline_accepts_several_scope_targets(map_ctx: ToolContext) -> None:
    result = await outline(
        map_ctx,
        {"scope": [{"document_id": DOC_A}, {"document_id": DOC_B}]},
    )
    assert result.error is None
    paths = {row["section_path"] for row in result.payload["sections"]}
    assert PATH_OVERVIEW in paths
    assert PATH_INTRO in paths
    assert "| document_id=" in result.text


@pytest.mark.asyncio
async def test_node_filter_keeps_ancestors_and_full_descendant_subtree(
    map_ctx: ToolContext,
) -> None:
    result = await node_filter(
        map_ctx,
        {
            "scope": [{"document_id": DOC_A}],
            "predicates": [{"field": "summary", "terms": ["hit-marker-findings"]}],
        },
    )
    assert result.error is None
    assert "[Hit]" in result.text
    assert PATH_FINDINGS in result.text
    assert PATH_ROOT in result.text
    assert PATH_OVERVIEW in result.text
    assert PATH_DETAIL in result.text
    assert PATH_TREATMENT not in result.text
    hits = result.payload["matched_sections"]
    assert [row["section_path"] for row in hits] == [PATH_FINDINGS]


@pytest.mark.asyncio
async def test_outline_folds_over_char_budget_and_shows_hidden_count(
    map_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.services.retrieval.agent_tools.tools.outline.MAP_TOOL_CHAR_BUDGET",
        80,
    )

    async def _scores(*_args: object, **_kwargs: object) -> dict[str, float]:
        return {
            "sec_overview": 10.0,
            "sec_findings": 10.0,
            "sec_detail": 10.0,
            "sec_treatment": 0.0,
        }

    monkeypatch.setattr(
        "shared.services.retrieval.agent_tools.tools.outline.load_leaf_unit_scores",
        _scores,
    )
    result = await outline(map_ctx, {"scope": [{"document_id": DOC_A}]})
    assert result.error is None
    assert "folded" in result.text
    assert "hidden" in result.text
    assert PATH_TREATMENT not in result.text or "hidden" in result.text


@pytest.mark.asyncio
async def test_outline_over_budget_without_query_fails(map_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.services.retrieval.agent_tools.tools.outline.MAP_TOOL_CHAR_BUDGET",
        80,
    )
    map_ctx.query = ""
    result = await outline(map_ctx, {"scope": [{"document_id": DOC_A}]})
    assert result.error is not None
    assert "no query" in result.error


@pytest.mark.asyncio
async def test_recall_passes_section_subtree_into_discovery(
    map_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def _discovery(*_args: object, **kwargs: object) -> DiscoveryResult:
        captured.update(kwargs)
        return DiscoveryResult(status="discovery_done", payload={"fused_rows": []})

    monkeypatch.setattr(
        "shared.services.retrieval.agent_tools.tools.recall.map_unit_discovery",
        _discovery,
    )
    result = await recall(
        map_ctx,
        {"query": "findings", "scope": [{"document_id": DOC_A, "section_path": PATH_OVERVIEW}]},
    )
    assert result.error is None
    assert captured["section_targets"] == [(DOC_A, PATH_OVERVIEW)]


@pytest.mark.asyncio
async def test_grep_subtree_scope_excludes_sibling_sections(map_ctx: ToolContext) -> None:
    result = await grep(
        map_ctx,
        {
            "pattern": "body",
            "scope": [{"document_id": DOC_A, "section_path": PATH_OVERVIEW}],
        },
    )
    assert result.error is None
    chunk_ids = [row["chunk_id"] for row in result.payload["results"]]
    assert "chunk_findings" in chunk_ids
    assert "chunk_detail" in chunk_ids
    assert "chunk_treatment" not in chunk_ids
    assert "| document_id=" in result.text


@pytest.mark.asyncio
async def test_assets_subtree_scope_keeps_only_connected_in_scope_assets(
    map_ctx: ToolContext,
) -> None:
    result = await assets(
        map_ctx,
        {"scope": [{"document_id": DOC_A, "section_path": PATH_OVERVIEW}], "type": "table"},
    )
    assert result.error is None
    chunk_ids = [row["chunk_id"] for row in result.payload["assets"]]
    assert chunk_ids == ["chunk_table_overview"]
    assert "chunk_table_treatment" not in result.text


def test_fold_at_map_tool_budget_hides_low_score_keeps_hit_chain() -> None:
    hit = MapNode("findings", "overview", render="H" * 8000, score=float("inf"))
    ancestor = MapNode("overview", "root", render="A" * 100, score=float("inf"))
    root = MapNode("root", None, render="R" * 50, score=float("inf"))
    low = MapNode("treatment", "root", render="L" * 15000, score=0.0)
    visible, hidden = fold_map_nodes(
        [root, ancestor, hit, low], char_budget=MAP_TOOL_CHAR_BUDGET
    )
    ids = [node.section_id for node in visible]
    assert "findings" in ids
    assert "overview" in ids
    assert "root" in ids
    assert "treatment" not in ids
    assert hidden["root"] >= 1
