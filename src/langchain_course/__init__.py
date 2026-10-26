from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

# START - Using custom tool by using Tavily Client

# tavily = TavilyClient()

# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches over the internet.
#     Args:
#         query: The query to search for.
#     Returns:
#         The search result.
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

# END - Using custom tool by using Tavily Client

class Source(BaseModel):
    """ Scheme for a source used by agent. """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """ Scheme for agent response with answer and sources. """

    answer:str = Field(description="The agent's response to query.")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-4.1-mini", 
                 temperature=0)

# We can also use Custom Function provided by Tavily which TavilySearch which is a Base tool,
# so we dont have to define a custom tool method and easily use TavilySearch()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs with AI ML skill set in Dallas, Texas")})
    print(result)
    #print(result.content)



if __name__ == '__main__' :
    main()
