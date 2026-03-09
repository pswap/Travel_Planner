"""Hotel booking agent node and hotel search for the travel planner graph."""

import json
import os
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import BaseTool, Tool

# Optional: SerpAPI for hotel search (serpapi or google-search-results)
try:
    import serpapi
    _SERPAPI_AVAILABLE = True
except ImportError:
    _SERPAPI_AVAILABLE = False


def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    hotel_class: Optional[str] = None,
    sort_by: int = 8,
) -> str:
    """
    Search for hotels using Google Hotels engine (SerpAPI).

    Args:
        location: Location to search (e.g. 'New York', 'Paris', 'Tokyo')
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of adults (default: 1)
        children: Number of children (default: 0)
        rooms: Number of rooms (default: 1)
        hotel_class: Hotel class filter (e.g. '2,3,4' for 2–4 star)
        sort_by: Sort option, e.g. 3=lowest price, 8=highest rating (default: 8)
    """
    if not _SERPAPI_AVAILABLE:
        return "Hotel search is unavailable: install 'serpapi' and set SERPAPI_API_KEY."

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Hotel search requires SERPAPI_API_KEY to be set in the environment."

    adults = int(float(adults)) if adults else 1
    children = int(float(children)) if children else 0
    rooms = int(float(rooms)) if rooms else 1
    sort_by = int(float(sort_by)) if sort_by else 8

    params = {
        "api_key": api_key,
        "engine": "google_hotels",
        "hl": "en",
        "gl": "us",
        "q": location,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "currency": "USD",
        "adults": adults,
        "children": children,
        "rooms": rooms,
        "sort_by": sort_by,
    }
    if hotel_class:
        params["hotel_class"] = hotel_class

    try:
        # New serpapi package: Client(api_key=...).search(...)
        if hasattr(serpapi, "Client"):
            client = serpapi.Client(api_key=api_key)
            result = client.search(**{k: v for k, v in params.items() if k != "api_key"})
            data = result if isinstance(result, dict) else getattr(result, "data", result)
        else:
            # Fallback: module-level search if available
            search = serpapi.search(params)
            data = getattr(search, "data", search) if not isinstance(search, dict) else search

        properties = data.get("properties", []) if isinstance(data, dict) else []
        if not properties:
            keys = list(data.keys()) if isinstance(data, dict) else []
            return f"No hotels found. Available keys: {keys}"

        return json.dumps(properties[:5], indent=2, default=str)
    except Exception as e:
        return f"Hotel search failed: {str(e)}"


def get_search_hotels_tool() -> Tool:
    """Return a LangChain Tool for search_hotels (for use with bind_tools)."""
    return Tool.from_function(
        func=search_hotels,
        name="search_hotels",
        description="Search for hotels using Google Hotels engine. Args: location, check_in_date (YYYY-MM-DD), check_out_date (YYYY-MM-DD), adults, children, rooms, hotel_class (optional), sort_by (e.g. 8=rating).",
    )


def create_hotel_agent_node(
    hotel_agent: RunnableSequence,
    tools: List[BaseTool],
):
    """Factory that returns the hotel agent node bound to the given agent and tools."""

    tools_by_name = {t.name: t for t in tools}

    def hotel_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Hotel booking agent node."""
        messages = state["messages"]
        response = hotel_agent.invoke({"messages": messages})

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            return {"messages": [response]}

        tool_messages = []
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {})
            print(f"🔧 Tool call: {name} | args: {args}")
            tool_call_id = tool_call["id"]
            try:
                tool = tools_by_name.get(name)
                if tool:
                    result = tool.invoke(args)
                    tool_result = json.dumps(result, indent=2, default=str) if isinstance(result, (dict, list)) else str(result)
                else:
                    tool_result = f"Unknown tool: {name}"
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"
            tool_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))

        all_messages = messages + [response] + tool_messages
        final_response = hotel_agent.invoke({"messages": all_messages})
        return {"messages": [response] + tool_messages + [final_response]}

    return hotel_agent_node
