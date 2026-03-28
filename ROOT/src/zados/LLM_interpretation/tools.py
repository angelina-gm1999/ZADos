"""
ZA-DOS LLM Interpretation Layer — tool definitions.

Defines the web_search tool in Ollama-compatible format and the
search execution stub.

Phase 1–3: _execute_search() returns a placeholder string.
Phase 4:   Replace the body with a real search provider
           (e.g. SearXNG, Brave Search API, DuckDuckGo).
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Tool definition (Ollama / OpenAI function-calling schema)
# ---------------------------------------------------------------------------

SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use when the user asks for facts, recent events, or data "
            "that may not be in training knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute.",
                }
            },
            "required": ["query"],
        },
    },
}

# Convenience list for passing directly to call_llama(tools=...)
SEARCH_TOOLS: List[Dict[str, Any]] = [SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

def _execute_search(query: str) -> str:
    """
    Execute a web search and return results as a plain string.

    Phase 1–3 stub — returns a placeholder so the LLM layer integrates
    cleanly before a live provider is wired in.

    Replace this body in Phase 4 with a real HTTP call to a search API.
    """
    # Phase 4 implementation note:
    #   resp = requests.get("https://api.search-provider.example/search",
    #                       params={"q": query, "key": API_KEY}, timeout=10)
    #   return resp.json()["results"][0]["snippet"]
    return f"[Search results for '{query}' — search provider not yet configured.]"
