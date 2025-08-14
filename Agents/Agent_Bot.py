from typing import TypedDict, List
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import os


load_dotenv()
BASE_URL: str = os.getenv("OPEN_AI_API_BASE", "")
API_KEY: SecretStr = SecretStr(os.getenv("OPEN_AI_API_KEY", ""))


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenAI(
    model="qwen3:0.6b",
    base_url=BASE_URL,
    api_key=API_KEY,
)


def process_node(state: AgentState) -> AgentState:
    """Simple process node, To get response from the LLM"""

    response = llm.invoke(state["messages"])
    print(f"AI => {response.content}")
    return state


PROCESS_NODE: str = "PROCESS"
graph = StateGraph(AgentState)
graph.add_node(PROCESS_NODE, process_node)
graph.add_edge(START, PROCESS_NODE)
graph.add_edge(PROCESS_NODE, END)

agent = graph.compile()

user_input = input("Enter => ")
while user_input != "quit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter => ")