"""Itinerary planning agent node for the travel planner graph."""

import json
from typing import Any, Callable, List

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import BaseTool


def create_itinerary_agent_node(
    itinerary_agent: RunnableSequence,
    tools: List[BaseTool],
):
    """Factory that returns the itinerary agent node bound to the given agent and tools."""

    tools_by_name = {t.name: t for t in tools}

    def itinerary_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Itinerary planning agent node."""
        messages = state["messages"]
        response = itinerary_agent.invoke({"messages": messages})

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
                    # If this was the retriever, result is a list of Documents
                    if name == "travel_planner_retriever" and isinstance(result, list):
                        sources = [getattr(d, "metadata", {}).get("source", "?") for d in result]
                        print(f"📚 Sources used: {sources}")
                        
                    if isinstance(result, (dict, list)):
                        tool_result = json.dumps(result, indent=2, default=str)
                    else:
                        tool_result = str(result)
                else:
                    tool_result = f"Unknown tool: {name}"
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"

            tool_messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call_id)
            )

        all_messages = messages + [response] + tool_messages
        final_response = itinerary_agent.invoke({"messages": all_messages})
        return {"messages": [response] + tool_messages + [final_response]}

    return itinerary_agent_node
