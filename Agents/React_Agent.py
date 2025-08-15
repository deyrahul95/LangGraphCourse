from pydantic import SecretStr
from typing import TypedDict, Sequence, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import os
import re


load_dotenv()
BASE_URL: str = os.getenv("OPEN_AI_API_BASE", "")
API_KEY: SecretStr = SecretStr(os.getenv("OPEN_AI_API_KEY", ""))
END_CONDITION: str = "END"
CONTINUE_CONDITION: str = "CONTINUE"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """This is a tool that adds 2 numbers together"""

    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """This is a tool that subtract 2nd number from 1st number"""

    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """This is a tool that multiply 2 numbers together"""

    return a * b


tools = [add, subtract, multiply]

llm = ChatOpenAI(
    model="qwen3:0.6b",
    base_url=BASE_URL,
    api_key=API_KEY,
).bind_tools(tools=tools)


def remove_think_block(text) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def process_node(state: AgentState) -> AgentState:
    """Simple process node, To get response from the LLM"""
    system_prompt = SystemMessage(
        content="You are my AI assistance, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])  # type: ignore
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:  # type: ignore
        return END_CONDITION
    else:
        return CONTINUE_CONDITION


state_graph = StateGraph(AgentState)

PROCESS_NODE: str = "PROCESS"
TOOL_NODE: str = "TOOL"
tool_node = ToolNode(tools=tools)

state_graph.add_node(PROCESS_NODE, process_node)
state_graph.add_node(TOOL_NODE, tool_node)

state_graph.add_edge(START, PROCESS_NODE)
state_graph.add_conditional_edges(
    PROCESS_NODE, should_continue, {CONTINUE_CONDITION: TOOL_NODE, END_CONDITION: END}
)
state_graph.add_edge(TOOL_NODE, PROCESS_NODE)

graph = state_graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = AgentState(messages=[HumanMessage(content="Add 20 and 30 then multiply the result with 10. Tell me a joke after that.")])
print_stream(graph.stream(input=inputs, stream_mode="values"))
