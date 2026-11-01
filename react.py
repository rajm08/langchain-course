from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple_tool(num:float) ->float :
    """
    This function takes in a input parameters and triples it.
    Param num: Input that is provided to triple.
    returns: Triple of the input number.
    """
    return float(num)*3

tools = [TavilySearch(max_results=1), triple_tool]

llm = ChatOpenAI(model="gpt-4-min", temperature=0).bind_tools(tools)
