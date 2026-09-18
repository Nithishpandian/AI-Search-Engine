# server/langchain_experiments/04_react_skeleton_langgraph.py
import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# --- 1. STATE ---
# This is the LangGraph equivalent of your `all_observations` scratchpad list.
# `add_messages` is a special reducer: instead of overwriting the list each time
# a node returns, it APPENDS new messages to it. This replaces the manual
# `all_observations.append(...)` calls you wrote by hand.
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int

# --- 2. FAKE TOOL ---
# A stand-in for your real search_web(). Deliberately trivial so we can
# verify the graph's routing logic without also debugging Tavily at the same time.
@tool
def fake_search(query: str) -> str:
    """Search the web for information about the query. Use this to find facts you don't already know."""
    print(f"  [fake_search called with query: '{query}']")
    return f"FAKE RESULT: '{query}' returned 3 mock sources about this topic. (This is not real data.)"

tools = [fake_search]

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)  # tool_choice="auto" by default — matches your V8's auto (not forced) choice

MAX_STEPS = 3  # same cap you used in production, lowered from 5

# --- 3. NODES ---
# Each node is a plain function: (state) -> partial state update.
# This one is your "Thought + Action" step: ask the LLM what to do next.
def agent_node(state: ResearchState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }

# This one is your "Observation" step: actually run whatever tool the LLM picked.
def tool_node(state: ResearchState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        result = fake_search.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages}

# --- 4. CONDITIONAL EDGE ---
# This replaces `if tool_call.name == "finish_research": break` from your
# hand-rolled loop. LangGraph calls this after every agent_node run to decide
# where to go next.
def should_continue(state: ResearchState) -> str:
    last_message = state["messages"][-1]

    if state["step_count"] >= MAX_STEPS:
        print(f"  [hit MAX_STEPS={MAX_STEPS}, forcing finish]")
        return "end"

    if last_message.tool_calls:
        return "continue"  # model wants to call a tool -> go run it
    return "end"  # model answered in plain text -> we're done

# --- 5. WIRE THE GRAPH ---
graph = StateGraph(ResearchState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
graph.add_edge("tools", "agent")  # after running a tool, always go back to the agent to think again

app = graph.compile()

# --- 6. RUNNING IT ---
initial_state = {
    "messages": [
        SystemMessage(content="You are a research assistant. Use the search tool if you need information, then give a final text answer when you have enough."),
        HumanMessage(content="What is the capital of France and what is its population?"),
    ],
    "step_count": 0,
}

final_state = app.invoke(initial_state)

print("\n--- FULL MESSAGE TRACE ---")
for msg in final_state["messages"]:
    print(f"[{msg.__class__.__name__}] {msg.content[:200] if msg.content else '(tool call, no content)'}")

print("\n--- FINAL ANSWER ---")
print(final_state["messages"][-1].content)
print(f"\nTotal steps: {final_state['step_count']}")