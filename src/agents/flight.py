"""Flight booking agent node and flight search logic for the travel planner graph."""

import json
import os
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.prompts.flight import FLIGHT_EXTRACTION_PROMPT

# Optional: SerpAPI for flight search
try:
    from serpapi.google_search import GoogleSearch
except ImportError:
    GoogleSearch = None

# Optional: dateutil for date parsing
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None


# --- Flight search parameters ---


class FlightSearchParams(BaseModel):
    """Structured flight search parameters extracted from user query."""

    departure_airport: str
    arrival_airport: str
    outbound_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0


AIRPORT_CODE_MAP = {
    "boston": "BOS",
    "san francisco": "SFO",
    "sf": "SFO",
    "new york": "JFK",
    "nyc": "JFK",
    "newark": "EWR",
    "los angeles": "LAX",
    "chicago": "ORD",
    "seattle": "SEA",
    "paris": "CDG",
    "london": "LHR",
    "delhi": "DEL",
    "tokyo": "NRT",
    "singapore": "SIN",
    "munich": "MUC",
    "nagpur": "NAG",
}


def normalize_airport_code(name: str) -> str:
    """Convert airport city names to IATA codes if possible."""
    if not name:
        return name
    name_lower = name.strip().lower()
    return AIRPORT_CODE_MAP.get(name_lower, name.upper() if len(name) <= 3 else name)


def normalize_date(date_str: str) -> str:
    """Convert various human date formats (e.g. 'Nov 29 2025') to 'YYYY-MM-DD'."""
    if not date_str:
        return date_str
    if date_parser is None:
        return date_str
    try:
        parsed = date_parser.parse(date_str)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return date_str


# --- Flight search (SerpAPI Google Flights) ---


def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
) -> str:
    """Search and format flight results via SerpAPI Google Flights."""
    if GoogleSearch is None:
        return "Flight search is unavailable: install 'google-search-results' and set SERPAPI_API_KEY."

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Flight search requires SERPAPI_API_KEY to be set in the environment."

    params = {
        "api_key": api_key,
        "engine": "google_flights",
        "departure_id": departure_airport,
        "arrival_id": arrival_airport,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "currency": "USD",
        "adults": adults,
        "children": children,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        flights = results.get("best_flights") or results.get("other_flights") or []
        if not flights:
            return f"No flights found for the given criteria. Raw results: {json.dumps(results, default=str)[:500]}"

        formatted = []
        for f in flights[:15]:
            main = f["flights"][0]
            airline = main.get("airline", "Unknown")
            dep = main["departure_airport"]["name"]
            arr = main["arrival_airport"]["name"]
            dep_time = main["departure_airport"]["time"]
            arr_time = main["arrival_airport"]["time"]
            duration = f.get("total_duration") or main.get("duration", "?")
            price = f.get("price", "N/A")
            cls = main.get("travel_class", "Unknown")
            formatted.append(
                f"- {airline} ({cls}) — ${price}, {duration} min\n"
                f"  {dep} ({dep_time}) → {arr} ({arr_time})"
            )
        return "✈️ **Top Flight Options:**\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Flight search failed: {str(e)}"


# --- Run flight agent (extract params → normalize → search) ---


def create_run_flight_agent(llm: Runnable) -> Callable[[str], str]:
    """Build a run_flight_agent callable that uses the given LLM for parameter extraction."""

    parser = PydanticOutputParser(pydantic_object=FlightSearchParams)
    flight_extraction_chain = FLIGHT_EXTRACTION_PROMPT | llm | parser

    def run_flight_agent(user_query: str) -> str:
        print(f"🧠 Extracting parameters for: '{user_query}'")
        try:
            params: FlightSearchParams = flight_extraction_chain.invoke({"user_query": user_query})
        except Exception as e:
            return f"Could not extract flight parameters from your message: {e}. Please specify departure and arrival (e.g. cities or airport codes) and travel date."

        print("✅ Extracted parameters:", params.model_dump_json(indent=2))

        params.departure_airport = normalize_airport_code(params.departure_airport)
        params.arrival_airport = normalize_airport_code(params.arrival_airport)
        params.outbound_date = normalize_date(params.outbound_date)
        if params.return_date:
            params.return_date = normalize_date(params.return_date)

        print(
            f"  Departure: {params.departure_airport}, Arrival: {params.arrival_airport}, "
            f"Outbound: {params.outbound_date}, Return: {params.return_date}"
        )
        print("🌍 Searching flights...")
        return search_flights(
            departure_airport=params.departure_airport,
            arrival_airport=params.arrival_airport,
            outbound_date=params.outbound_date,
            return_date=params.return_date,
            adults=params.adults,
            children=params.children,
        )

    return run_flight_agent


# --- LangChain tool (optional, for binding to LLM) ---


def get_search_flights_tool():
    """Return a LangChain Tool for search_flights (for use with bind_tools if needed)."""
    from langchain_core.tools import Tool

    return Tool.from_function(
        func=search_flights,
        name="search_flights",
        description="Search for flights using Google Flights via SerpAPI.",
    )


# --- Graph node factory ---


def _default_run_flight_agent(query: str) -> str:
    """Stub when no flight search implementation is provided."""
    return "Flight search is not yet implemented. Pass run_flight_agent from create_run_flight_agent(llm) to enable it."


def create_flight_agent_node(
    run_flight_agent: Callable[[str], str] | None = None,
):
    """Factory that returns the flight agent node.

    Args:
        run_flight_agent: Callable that takes the user query and returns a formatted
            flight search result string. Use create_run_flight_agent(llm) to create one.
    """
    run_flight = run_flight_agent or _default_run_flight_agent

    def flight_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Flight booking agent node."""
        messages = state["messages"]
        user_query = messages[-1].content
        tool_result = run_flight(user_query)
        return {"messages": [HumanMessage(content=tool_result)]}

    return flight_agent_node
