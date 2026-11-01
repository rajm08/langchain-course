from dotenv import load_dotenv

from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use the tools to answer the questions.
"""

def run_agent_reasoning(state:MessagesState) -> MessagesState:
    """
    Run the agent reasonining node.
    """

    response = llm.invoke([{"role":"system", "content":SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages":[response]}


tool_node = ToolNode(tools)