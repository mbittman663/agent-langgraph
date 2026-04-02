################# Agent creation backend #################

from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# =========================
# 0. Load environment vars
# =========================

load_dotenv()  # loads OPENAI_API_KEY from .env

# =========================
# 1. Define State
# =========================

class AgentState(TypedDict):
    input: str
    plan: str
    research: str
    output: str
    history: List[str]


# =========================
# 2. Initialize LLM
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================
# 3. Tools
# =========================

def search_tool(query: str) -> str:
    # Replace this with real API later
    return f"[Search results for: {query}]"


# =========================
# 4. Agent Nodes
# =========================

def planner(state: AgentState) -> AgentState:
    prompt = f"""
    Break this task into a clear step-by-step plan:

    Task:
    {state['input']}
    """

    response = llm.invoke(prompt).content

    state["plan"] = response
    state["history"].append(f"PLAN:\n{response}")
    return state


def researcher(state: AgentState) -> AgentState:
    # simulate tool usage
    tool_result = search_tool(state["plan"])

    prompt = f"""
    Use the following plan and data to gather insights:

    Plan:
    {state['plan']}

    Data:
    {tool_result}
    """

    response = llm.invoke(prompt).content

    state["research"] = response
    state["history"].append(f"RESEARCH:\n{response}")
    return state


def writer(state: AgentState) -> AgentState:
    prompt = f"""
    Write a clean, well-structured final answer based on:

    {state['research']}
    """

    response = llm.invoke(prompt).content

    state["output"] = response
    state["history"].append(f"OUTPUT:\n{response}")
    return state


# =========================
# 5. Build Graph
# =========================

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")

    return builder.compile()


# =========================
# 6. Run App
# =========================

def run_agent(user_input: str):
    graph = build_graph()

    initial_state: AgentState = {
        "input": user_input,
        "plan": "",
        "research": "",
        "output": "",
        "history": []
    }

    result = graph.invoke(initial_state)

    return result


# =========================
# 7. CLI Entry Point
# =========================

if __name__ == "__main__":
    user_input = input("Enter your task: ")

    result = run_agent(user_input)

    print("\n=== FINAL OUTPUT ===\n")
    print(result["output"])

    print("\n=== DEBUG HISTORY ===\n")
    for step in result["history"]:
        print(step)
        print("-" * 50)
