"""Utilities for adapting MCP Servers to simple async call patterns used in tests."""

from __future__ import annotations

from typing import Any, Dict, List

import mcp.types as mcp_types
from mcp.server import Server


def attach_direct_methods(server: Server) -> None:
    """Attach awaitable `list_tools` and `call_tool` helpers to a Server instance.

    The low-level MCP Server API uses decorators to register handlers. These
    helpers provide simple coroutine wrappers so tests can call
    `await server.list_tools()` and `await server.call_tool(name, args)` without
    constructing request objects.
    """

    async def list_tools_direct() -> List[mcp_types.Tool]:
        handler = server.request_handlers.get(mcp_types.ListToolsRequest)
        if handler is None:
            return []

        response = await handler(mcp_types.ListToolsRequest())
        root = getattr(response, "root", None)
        return list(getattr(root, "tools", []) or [])

    async def call_tool_direct(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
        handler = server.request_handlers.get(mcp_types.CallToolRequest)
        if handler is None:
            return []

        request = mcp_types.CallToolRequest(
            params=mcp_types.CallToolRequestParams(name=name, arguments=arguments)
        )
        response = await handler(request)
        root = getattr(response, "root", None)
        if root and hasattr(root, "content"):
            return list(root.content)
        return []

    # Override instance attributes (safe for test usage)
    server.list_tools = list_tools_direct  # type: ignore[attr-defined,assignment]
    server.call_tool = call_tool_direct  # type: ignore[attr-defined,assignment]




