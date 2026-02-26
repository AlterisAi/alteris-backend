"""Loom MCP Server — HTTP/SSE transport on localhost.

Exposes the Knowledge Graph, Clarity Queue, Agent system, and Pipeline
as MCP tools. Serves as the universal interface for:
  - The Swift macOS app (replacing CLI subprocess calls for data queries)
  - User-created agents (sandboxed, read-only KG access)
  - Claude Desktop / other MCP clients

Usage:
    python -m loom.mcp_server
    python -m loom.mcp_server --port 9119
    python -m loom.mcp_server --read-only  # Only expose read tools (for agents)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from loom.constants import MCP_DEFAULT_PORT, MCP_HOST
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def create_server(
    store: LayeredGraphStore,
    read_only: bool = False,
):
    """Create and configure the MCP server with all tools.

    Args:
        store: The LayeredGraphStore instance to query.
        read_only: If True, only register 'read' tools (for user agents).

    Returns:
        A configured MCP server instance.
    """
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("loom-kg")

    # Load all tool definitions
    from loom.mcp_tools import ensure_tools_loaded, get_all_tools, get_read_tools
    ensure_tools_loaded()

    tools = get_read_tools() if read_only else get_all_tools()
    tool_map = {t.name: t for t in tools}

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return all available tools."""
        result = []
        for t in tools:
            # Build JSON Schema for parameters
            properties = {}
            required = []
            for p in t.params:
                prop: dict = {"type": p.type, "description": p.description}
                if p.enum:
                    prop["enum"] = p.enum
                if p.default is not None:
                    prop["default"] = p.default
                properties[p.name] = prop
                if p.required:
                    required.append(p.name)

            schema = {
                "type": "object",
                "properties": properties,
            }
            if required:
                schema["required"] = required

            result.append(Tool(
                name=t.name,
                description=t.description,
                inputSchema=schema,
            ))
        return result

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Execute a tool and return results."""
        tool_def = tool_map.get(name)
        if not tool_def:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        if not tool_def.handler:
            return [TextContent(type="text", text=json.dumps({"error": f"Tool {name} has no handler"}))]

        try:
            result = tool_def.handler(store, **arguments)
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc, exc_info=True)
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return server


async def run_sse_server(port: int, store: LayeredGraphStore, read_only: bool = False):
    """Run the MCP server with SSE transport."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse
    import uvicorn

    server = create_server(store, read_only=read_only)
    sse = SseServerTransport("/messages/")

    # Build tool lookup for the /rpc endpoint
    from loom.mcp_tools import ensure_tools_loaded, get_all_tools, get_read_tools
    ensure_tools_loaded()
    tools = get_read_tools() if read_only else get_all_tools()
    tool_map = {t.name: t for t in tools}

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    # Health check endpoint
    async def health(request):
        stats = store.stats()
        return JSONResponse({
            "status": "ok",
            "read_only": read_only,
            "events": stats.get("events_count", 0),
            "beliefs": stats.get("beliefs_count", 0),
            "persons": stats.get("persons_count", 0),
        })

    # Direct JSON-RPC endpoint for the Swift app (simpler than full SSE)
    async def handle_rpc(request):
        from starlette.responses import Response
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})

        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if method == "tools/list":
            tool_list = []
            for t in tools:
                properties = {}
                required_params = []
                for p in t.params:
                    prop: dict = {"type": p.type, "description": p.description}
                    if p.enum:
                        prop["enum"] = p.enum
                    if p.default is not None:
                        prop["default"] = p.default
                    properties[p.name] = prop
                    if p.required:
                        required_params.append(p.name)
                schema = {"type": "object", "properties": properties}
                if required_params:
                    schema["required"] = required_params
                tool_list.append({"name": t.name, "description": t.description, "inputSchema": schema})
            return JSONResponse({"jsonrpc": "2.0", "result": {"tools": tool_list}, "id": req_id})

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            tool_def = tool_map.get(tool_name)
            if not tool_def:
                return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}, "id": req_id})
            if not tool_def.handler:
                return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Tool {tool_name} has no handler"}, "id": req_id})
            try:
                result = tool_def.handler(store, **arguments)
                text = json.dumps(result, default=str)
                return JSONResponse({"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": text}]}, "id": req_id})
            except Exception as exc:
                logger.error("RPC tool %s failed: %s", tool_name, exc, exc_info=True)
                return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(exc)}, "id": req_id})

        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown method: {method}"}, "id": req_id})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/rpc", handle_rpc, methods=["POST"]),
            Route("/sse", handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    config = uvicorn.Config(
        app,
        host=MCP_HOST,
        port=port,
        log_level="info",
    )
    srv = uvicorn.Server(config)
    logger.info("MCP server starting on %s:%d (read_only=%s)", MCP_HOST, port, read_only)
    await srv.serve()


async def run_stdio_server(store: LayeredGraphStore, read_only: bool = False):
    """Run the MCP server with stdio transport (for Claude Desktop integration)."""
    from mcp.server.stdio import stdio_server

    server = create_server(store, read_only=read_only)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        prog="loom-mcp",
        description="Loom MCP Server — Knowledge Graph interface",
    )
    parser.add_argument(
        "--port", type=int, default=MCP_DEFAULT_PORT,
        help=f"Port to listen on (default: {MCP_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Database path (default: ~/.loom/graph.db)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="Only expose read tools (for user agents)",
    )
    parser.add_argument(
        "--stdio", action="store_true",
        help="Use stdio transport instead of HTTP/SSE (for Claude Desktop)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    store = LayeredGraphStore(args.db_path)

    import asyncio

    if args.stdio:
        asyncio.run(run_stdio_server(store, read_only=args.read_only))
    else:
        asyncio.run(run_sse_server(args.port, store, read_only=args.read_only))


if __name__ == "__main__":
    main()
