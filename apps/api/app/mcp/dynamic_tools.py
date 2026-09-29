"""Bridge ``shared.services.retrieval.agent_tools.REGISTRY`` onto FastMCP.

FastMCP's public registration API (``FastMCP.tool`` / ``ToolManager.add_tool``)
only builds a tool's schema by introspecting a Python function's *signature*
(``mcp.server.fastmcp.tools.base.Tool.from_function`` -> ``func_metadata``);
there is no public entry point to register a tool from an already-built JSON
Schema dict, which is what every ``agent_tools.ToolSpec`` carries. Since our
schema is the one already shipped to ``agent_explore`` and meant to be
verbatim-identical across harnesses (see ``CORPUS_SCHEMA.md``), we construct
``Tool`` objects directly instead of round-tripping through a synthetic
Python function signature, and insert them into the tool manager's registry
dict — the same dict ``ToolManager.__init__`` accepts a ``tools=`` list for,
just with no public single-tool equivalent of that constructor path.
"""

from __future__ import annotations

from typing import Any, AsyncContextManager, Callable

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from pydantic import ConfigDict, create_model
from sqlalchemy.ext.asyncio import AsyncSession

from shared.services.retrieval.agent_tools import REGISTRY, ToolContext, ToolSpec

DbFactory = Callable[[], AsyncContextManager[AsyncSession]]


class _AllowExtraArgModel(ArgModelBase):
    """``ArgModelBase`` that also forwards unrecognized keys.

    ``FuncMetadata.call_fn_with_arg_validation`` builds ``_dispatch_tool``'s
    kwargs by calling ``model_dump_one_level()``, which only iterates this
    class's *declared* ``model_fields`` — verified by reading
    ``mcp.server.fastmcp.utilities.func_metadata`` directly. Declaring
    ``extra="allow"`` alone is not enough: pydantic stores unrecognized keys
    on the instance (``model_extra``) but the base implementation never
    reads them. Overriding ``model_dump_one_level`` to merge them in is what
    actually gets an unknown key from an MCP client into ``_dispatch_tool``'s
    ``kwargs`` -> ``REGISTRY.dispatch`` -> ``validate_tool_args``'s unified
    unknown-argument error, instead of the previous fully-permissive model
    (every field ``(Any, None)``, no ``extra`` declared) silently dropping
    it before dispatch ever saw it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def model_dump_one_level(self) -> dict[str, Any]:
        dumped = super().model_dump_one_level()
        dumped.update(self.model_extra or {})
        return dumped


def _lenient_arg_model(spec: ToolSpec) -> type[ArgModelBase]:
    """Build a pydantic arg model for FastMCP's internal validation.

    The schema actually exposed to MCP clients is ``spec.json_schema``
    (``Tool.parameters``, returned verbatim in ``tools/list`` — see
    ``FastMCP.list_tools``); this model only has to satisfy
    ``FuncMetadata.call_fn_with_arg_validation`` well enough to forward
    whatever the client sent through to ``_dispatch_tool``'s ``**kwargs``,
    unknown keys included (``_AllowExtraArgModel`` above) — real
    argument-shape enforcement happens once in
    ``agent_tools.registry.validate_tool_args``, not duplicated here.
    Every declared property is optional/``Any``: each tool already
    validates its own required args through that same unified path.
    """
    properties = spec.json_schema.get("properties", {})
    fields: dict[str, Any] = {key: (Any, None) for key in properties}
    return create_model(
        f"{spec.name.replace('.', '_')}_Args", __base__=_AllowExtraArgModel, **fields
    )


def _make_tool(spec: ToolSpec, *, db_factory: DbFactory) -> Tool:
    async def _dispatch_tool(
        ctx: Context | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from app.mcp.retrieval_server import resolve_mcp_namespace, resolve_mcp_user_id

        namespace = resolve_mcp_namespace(ctx=ctx)
        async with db_factory() as db:
            user_id = await resolve_mcp_user_id(ctx=ctx, db=db)
            tool_ctx = ToolContext(
                db=db, user_id=user_id, namespace=namespace, db_factory=db_factory
            )
            result = await REGISTRY.dispatch(spec.name, tool_ctx, kwargs)
        return {"text": result.text, "payload": result.payload, "refs": result.refs, "error": result.error}

    return Tool(
        fn=_dispatch_tool,
        name=spec.name,
        title=None,
        description=spec.description,
        parameters=spec.json_schema,
        fn_metadata=FuncMetadata(arg_model=_lenient_arg_model(spec)),
        is_async=True,
        context_kwarg="ctx",
        annotations=None,
    )


def register_corpus_tools(server: FastMCP, *, db_factory: DbFactory) -> None:
    """Register every ``agent_tools.REGISTRY`` tool onto ``server``."""
    for spec in REGISTRY.all():
        tool = _make_tool(spec, db_factory=db_factory)
        server._tool_manager._tools[tool.name] = tool
