"""Contract coverage for immutable canonical demo artifact reuse."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
from httpx import AsyncClient
from pytest import MonkeyPatch

from tests.support.contract_database import ContractDatabase


def _write_source(directory: Path, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.md").write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_canonical_bundle_reuses_and_versions_real_filesystem_objects(
    api_client_factory: Callable[[], AbstractAsyncContextManager[AsyncClient]],
    tmp_path: Path,
) -> None:
    async with api_client_factory():
        from app.services.demo.canonical_bundle import CanonicalDemoBundleStore
        from shared.services.redis import RedisServiceFactory

        source_directory = tmp_path / "source"
        _write_source(source_directory, "version one")
        store = CanonicalDemoBundleStore(
            redis_service=RedisServiceFactory.get_service()
        )

        first = await store.ensure_bundle(
            source_id="contract-source",
            source_directory=source_directory,
        )
        reused = await store.ensure_bundle(
            source_id="contract-source",
            source_directory=source_directory,
        )

        assert first.reused is False
        assert reused.reused is True
        assert reused.content_version == first.content_version
        assert reused.zip_key == first.zip_key
        assert reused.raw_prefix == first.raw_prefix

        _write_source(source_directory, "version two")
        changed = await store.ensure_bundle(
            source_id="contract-source",
            source_directory=source_directory,
        )

    assert changed.reused is False
    assert changed.content_version != first.content_version
    assert changed.zip_key != first.zip_key
    assert changed.raw_prefix != first.raw_prefix


@pytest.mark.asyncio
async def test_canonical_bundle_retries_when_completion_marker_is_missing(
    api_client_factory: Callable[[], AbstractAsyncContextManager[AsyncClient]],
    tmp_path: Path,
) -> None:
    async with api_client_factory():
        from app.services.demo.canonical_bundle import CanonicalDemoBundleStore
        from shared.services.redis import RedisServiceFactory

        source_directory = tmp_path / "source"
        _write_source(source_directory, "marker recovery")
        store = CanonicalDemoBundleStore(
            redis_service=RedisServiceFactory.get_service()
        )
        initial = await store.ensure_bundle(
            source_id="contract-marker-recovery",
            source_directory=source_directory,
        )
        marker_key = f"{initial.raw_prefix}.bundle-ready.json"
        assert store._files.storage_adapter.delete_object(
            marker_key, store._files.results_bucket
        )

        original_writer = store._write_ready_marker

        def fail_marker(bundle: object) -> None:
            raise RuntimeError("marker write failed")

        store._write_ready_marker = fail_marker  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="marker write failed"):
            await store.ensure_bundle(
                source_id="contract-marker-recovery",
                source_directory=source_directory,
            )

        store._write_ready_marker = original_writer  # type: ignore[method-assign]
        recovered = await store.ensure_bundle(
            source_id="contract-marker-recovery",
            source_directory=source_directory,
        )

    assert recovered.reused is False
    assert recovered.zip_key == initial.zip_key
    assert recovered.content_version == initial.content_version


@pytest.mark.asyncio
async def test_canonical_bundle_concurrent_callers_publish_once(
    api_client_factory: Callable[[], AbstractAsyncContextManager[AsyncClient]],
    tmp_path: Path,
) -> None:
    async with api_client_factory():
        from app.services.demo.canonical_bundle import CanonicalDemoBundleStore
        from shared.services.redis import RedisServiceFactory

        source_directory = tmp_path / "source"
        _write_source(source_directory, "concurrent publication")
        redis_service = RedisServiceFactory.get_service()
        first_store = CanonicalDemoBundleStore(redis_service=redis_service)
        second_store = CanonicalDemoBundleStore(redis_service=redis_service)

        results = await asyncio.gather(
            first_store.ensure_bundle(
                source_id="contract-concurrent",
                source_directory=source_directory,
            ),
            second_store.ensure_bundle(
                source_id="contract-concurrent",
                source_directory=source_directory,
            ),
        )

    assert len({result.zip_key for result in results}) == 1
    assert len({result.raw_prefix for result in results}) == 1
    assert sum(not result.reused for result in results) == 1


@pytest.mark.asyncio
async def test_http_materialization_reuses_canonical_bundle_across_namespaces(
    developer_api_client_factory: Callable[
        [], AbstractAsyncContextManager[AsyncClient]
    ],
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CANONICAL_BUNDLE_ENABLED", "true")
    async with developer_api_client_factory() as api_client:
        import app.services.demo.source_materializer as source_materializer_module

        first = await api_client.post(
            "/api/v1/demo/materializations",
            json={
                "namespace": "canonical-contract-one",
                "demo_source_ids": ["demo-tsla-q4-2025"],
            },
        )
        second = await api_client.post(
            "/api/v1/demo/materializations",
            json={
                "namespace": "canonical-contract-two",
                "demo_source_ids": ["demo-tsla-q4-2025"],
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        first_document_id = str(first.json()["sources"][0]["document_id"])
        second_document_id = str(second.json()["sources"][0]["document_id"])
        chunks_response = await api_client.get(
            f"/api/v1/documents/{first_document_id}/chunks?page_size=200"
            "&include_asset_urls=true"
        )
        retrieval_response = await api_client.post(
            "/api/v1/retrieval/query",
            json={
                "namespace": "canonical-contract-one",
                "query": "Tesla investment",
                "top_k": 3,
                "use_agentic": False,
            },
        )
        assert retrieval_response.status_code == 200
        retrieval_asset_urls: list[str] = [
            str(result["asset_url"])
            for result in retrieval_response.json()["results"]
            if result.get("asset_url")
        ]
        assert retrieval_asset_urls
        assert all("demo-canonical" in url for url in retrieval_asset_urls)
        rows = await ContractDatabase.fetch_all(
            """
            SELECT d.document_id, jr.result_s3_key, jr.result_size,
                   jr.document_metadata ->> 'result_raw_prefix' AS result_raw_prefix
            FROM job_results AS jr
            JOIN documents AS d ON d.current_job_result_id = jr.id
            WHERE d.document_id IN (:first_document_id, :second_document_id)
            ORDER BY d.document_id
            """,
            {
                "first_document_id": first_document_id,
                "second_document_id": second_document_id,
            },
        )

        monkeypatch.setattr(
            source_materializer_module.settings,
            "DEMO_CANONICAL_BUNDLE_ENABLED",
            False,
        )
        existing_document_response = await api_client.get(
            f"/api/v1/documents/{first_document_id}/chunks?page_size=200"
            "&include_asset_urls=true"
        )

    chunks_body = chunks_response.json()
    asset_urls = [
        str(chunk["asset_url"])
        for chunk in chunks_body["chunks"]
        if chunk.get("asset_url")
    ]
    assert asset_urls
    assert any("demo-canonical" in asset_url for asset_url in asset_urls)
    assert existing_document_response.status_code == 200
    existing_asset_urls = [
        str(chunk["asset_url"])
        for chunk in existing_document_response.json()["chunks"]
        if chunk.get("asset_url")
    ]
    assert existing_asset_urls == asset_urls
    assert len(rows) == 2
    assert len({str(row["result_s3_key"]) for row in rows}) == 1
    assert len({str(row["result_raw_prefix"]) for row in rows}) == 1
    assert all(int(row["result_size"]) > 0 for row in rows)
