"""
DAiW Music-Brain MCP Server Package

Unified MCP server for music production tools.
"""

__version__ = "1.0.0"

# Export helpers for creating/running the MCP server.
# Alias run_server to the stdio implementation for backwards compatibility.
from daiw_mcp.server import create_server, run_server_stdio

run_server = run_server_stdio

__all__ = [
    "create_server",
    "run_server",
    "run_server_stdio",
]
