"""
Main entry point for the Travel Planner application.
"""
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from src.agents import (
    create_flight_agent_node,
    create_hotel_agent_node,
    create_itinerary_agent_node,
    create_run_flight_agent,
    get_search_hotels_tool,
)
from src.prompts.itinerary import ITINERARY_PROMPT
from src.prompts.hotel import HOTEL_PROMPT
from src.vectorstores.pinecone_store import PineconeVectorStore


# Load API key
load_dotenv()


def main():
    """Main function to run the application."""
    print("Welcome to Travel Planner!")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in .env")

    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=EMBEDDING_MODEL_NAME)
    EMBEDDING_DIMENSION = 1536
    print(f"Using OpenAI embedding model.")
    print(f"Successfully initialized Pinecone client and embedding model: {EMBEDDING_MODEL_NAME} (Dim: {EMBEDDING_DIMENSION}).")

    # 1. Load, chunk and index the contents of the blog to create a retriever.
    loader = WebBaseLoader(web_paths=("https://japanstartshere.com/one-week-in-japan/","https://girleatworld.net/japan-itinerary/"))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Loaded {len(splits)} chunks from the web page.")

    # 2. Create a Pinecone vector store and add the splits to it.
    # Setup Pinecone index
    pc = Pinecone()
    INDEX_NAME = "travel-planner-notes"
    METRIC = "cosine"
    try:
        import time

        existing_index_names = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_index_names:
            print(f"Index '{INDEX_NAME}' does not exist. Creating new serverless index...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while True:
                desc = pc.describe_index(INDEX_NAME)
                status = getattr(desc, "status", None) or {}
                ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
                if ready:
                    break
                print("Waiting for index to be ready...")
                time.sleep(5)
            print(f"Successfully created index '{INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}.")
        else:
            print(f"Index '{INDEX_NAME}' already exists.")

        pinecone_index_obj = pc.Index(INDEX_NAME)
        print(f"Successfully connected to index '{INDEX_NAME}'.")
        stats = pinecone_index_obj.describe_index_stats()
        print(f"Index stats: {stats}")

        # Vector store for RAG (e.g. add_documents, similarity_search)
        vectorstore = PineconeVectorStore(
            index=pinecone_index_obj,
            embedding=embeddings,
            text_key="text",
        )
        vectorstore.add_documents(splits)
        print(f"Successfully added {len(splits)} chunks to the vector store.")
    except Exception as e:
        print(f"Error during Pinecone index setup for '{INDEX_NAME}': {str(e)}")
        raise

    pinecone_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("Successfully created retriever (k=4).")

    pinecone_retriever_tool = Tool(
        name="travel_planner_retriever",
        description="A tool for retrieving travel information from the vector store.",
        func=pinecone_retriever.invoke,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    tavilySearchtool = TavilySearch(max_results=2)
    tools = [tavilySearchtool, pinecone_retriever_tool]
    # Test TavilySearch
    # result = tool.invoke("What is the weather in SF right now?")
    # print(result)

    # Test pinecone_retriever_tool
    # result = pinecone_retriever_tool.invoke("What is the weather in SF right now?")
    # print(result)

   

    def create_router():
        """Creates a router for the three travel agents using LangGraph patterns"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing expert for a travel planning system.

            Analyze the user's query and decide which specialist agent should handle it:

            - FLIGHT: Flight bookings, airlines, air travel, flight search, tickets, airports, departures, arrivals, airline prices
            - HOTEL: Hotels, accommodations, stays, rooms, hotel bookings, lodging, resorts, hotel search, hotel prices
            - ITINERARY: Travel itineraries, trip planning, destinations, activities, attractions, sightseeing, travel advice, weather, culture, food, general travel questions

            Respond with ONLY one word: FLIGHT, HOTEL, or ITINERARY

            Examples:
            "Book me a flight to Paris" → FLIGHT
            "Find hotels in Tokyo" → HOTEL
            "Plan my 5-day trip to Italy" → ITINERARY
            "Search flights from NYC to London" → FLIGHT
            "Where should I stay in Bali?" → HOTEL
            "What are the best attractions in Rome?" → ITINERARY
            "I need airline tickets" → FLIGHT
            "Show me hotel options" → HOTEL
            "Create an itinerary for Japan" → ITINERARY"""),

            ("user", "Query: {query}")
        ])

        router_chain = router_prompt | llm | StrOutputParser()
        def route_query(state):
            """Router function for LangGraph - decides which agent to call next"""

            # Get the latest user message
            user_message = state["messages"][-1].content

            print(f"🧭 Router analyzing: '{user_message[:50]}...'")

            try:
                # Get LLM routing decision
                decision = router_chain.invoke({"query": user_message}).strip().upper()

                # Map to our agent node names
                agent_mapping = {
                    "FLIGHT": "flight_agent",
                    "HOTEL": "hotel_agent",
                    "ITINERARY": "itinerary_agent"
                }

                next_agent = agent_mapping.get(decision, "itinerary_agent")
                print(f"🎯 Router decision: {decision} → {next_agent}")

                return next_agent

            except Exception as e:
                print(f"⚠️ Router error, defaulting to itinerary_agent: {e}")
                return "itinerary_agent"

        return route_query

    # Create the router
    router = create_router()
    print("✅ Travel Router created for LangGraph!")

    # Define state schema for travel multiagent system
    class TravelPlannerState(TypedDict):
        """Simple state schema for travel multiagent system"""

        # Conversation history - persisted with checkpoint memory
        messages: Annotated[List[BaseMessage], operator.add]

        # Agent routing
        next_agent: Optional[str]

        # Current user query
        user_query: Optional[str]

    def router_node(state: TravelPlannerState):
        """Router node - determines which agent should handle the query"""
        user_message = state["messages"][-1].content
        next_agent = router(state)

        return {
            "next_agent": next_agent,
            "user_query": user_message
        }

    # Bind tools to the llm
    llm_with_tools = llm.bind_tools(tools)

    # Create itinerary agent
    itinerary_agent = ITINERARY_PROMPT | llm_with_tools
    # llm_with_tools.invoke(itenary_prompt)
    # You can now invoke itinerary_agent with messages when wiring the app UI.
    # Create itinerary agent node with bound agent and tools
    itinerary_agent_node = create_itinerary_agent_node(itinerary_agent, tools)

    # Create flight agent node with parameter extraction + SerpAPI flight search
    run_flight_agent = create_run_flight_agent(llm)
    flight_agent_node = create_flight_agent_node(run_flight_agent)

    # Create hotel agent node with HOTEL_PROMPT and search_hotels tool
    search_hotels_tool = get_search_hotels_tool()
    hotel_tools = [search_hotels_tool]
    llm_with_hotel_tools = llm.bind_tools(hotel_tools)
    hotel_agent = HOTEL_PROMPT | llm_with_hotel_tools
    hotel_agent_node = create_hotel_agent_node(hotel_agent, hotel_tools)

    # Conditional routing function
    def route_to_agent(state: TravelPlannerState):
        """Conditional edge function - routes to appropriate agent based on router decision"""

        next_agent = state.get("next_agent")

        if next_agent == "flight_agent":
            return "flight_agent"
        elif next_agent == "hotel_agent":
            return "hotel_agent"
        elif next_agent == "itinerary_agent":
            return "itinerary_agent"
        else:
            # Default fallback
            return "itinerary_agent"

    # Build the complete travel planning graph
    workflow = StateGraph(TravelPlannerState)

    # Add all nodes to the graph
    workflow.add_node("router", router_node)
    workflow.add_node("flight_agent", flight_agent_node)
    workflow.add_node("hotel_agent", hotel_agent_node)
    workflow.add_node("itinerary_agent", itinerary_agent_node)

    # Set entry point - always start with router
    workflow.set_entry_point("router")

    # Add conditional edge from router to appropriate agent
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itinerary_agent": "itinerary_agent"
        }
    )

    # Add edges from each agent back to END
    workflow.add_edge("flight_agent", END)
    workflow.add_edge("hotel_agent", END)
    workflow.add_edge("itinerary_agent", END)

    checkpointer = InMemorySaver()

    # Compile the graph
    travel_planner = workflow.compile(checkpointer=checkpointer)

    print("✅ Travel Planning Graph built successfully!")

    # Test prompts (uncomment one to use, or copy for multi-turn chat)
    # test_prompt = "Plan a one-week trip to Japan based on your travel notes."
    # test_prompt = "What is the weather in SF right now?"
    # test_prompt = "I need to book a flight from BOS to CDG, from Oct 28 to October 31 for year 2026 for 1 person"
    # test_prompt = "I need to book a flight from SFO to NRT, from Oct 28 to October 31 for year 2026 for 1 person. Plan a one-week trip to Japan based on your travel notes along with flight and hotel recommendations."
    # test_prompt = "I need to book a flight from NYC to SFO, from Oct 28 to October 31 for year 2026 for 1 person. Plan a one-week trip to SFO based on your travel notes along with flight and hotel recommendations."
    # result = travel_planner.invoke(
    #     {"messages": [HumanMessage(content=test_prompt)]},
    #     config={"configurable": {"thread_id": "once"}},
    # )
    # last = result["messages"][-1]
    # print(f"Assistant: {last.content}")

    def run_multi_turn_chat(compiled_graph):
        """Multi-turn conversation with checkpoint memory."""
        print("\n💬 Multi-Agent Travel Assistant (type 'quit' to exit)")
        print("=" * 50)
        config = {"configurable": {"thread_id": "1"}}
        while True:
            user_input = input("\n🧑 You: ").strip()
            if not user_input or user_input.lower() == "quit":
                break
            print("\n📊 Processing query...")
            result = compiled_graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config,
            )
            last = result["messages"][-1]
            response = getattr(last, "content", str(last)) or "(No response)"
            print(f"\n🤖 Assistant: {response}")
            print("-" * 50)
        print("Goodbye!")

    run_multi_turn_chat(travel_planner)


if __name__ == "__main__":
    main()
