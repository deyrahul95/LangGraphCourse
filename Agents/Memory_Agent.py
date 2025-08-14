from typing import TypedDict, List, Union
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import os


load_dotenv()
BASE_URL: str = os.getenv("OPEN_AI_API_BASE", "")
API_KEY: SecretStr = SecretStr(os.getenv("OPEN_AI_API_KEY", ""))


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(
    model="qwen3:0.6b",
    base_url=BASE_URL,
    api_key=API_KEY,
)


def process_node(state: AgentState) -> AgentState:
    """Simple process node, To get response from the LLM"""

    response = llm.invoke(state["messages"])
    print(f"\n AI => {response.content}")

    state["messages"].append(AIMessage(content=response.content))
    print(f"\n CURRENT STATE => {state['messages']}")

    return state


PROCESS_NODE: str = "PROCESS"
graph = StateGraph(AgentState)
graph.add_node(PROCESS_NODE, process_node)
graph.add_edge(START, PROCESS_NODE)
graph.add_edge(PROCESS_NODE, END)

agent = graph.compile()

conversion_history = []

user_input = input(" Enter => ")
while user_input != "quit":
    conversion_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversion_history})
    conversion_history = result["messages"]
    user_input = input("\n Enter => ")
