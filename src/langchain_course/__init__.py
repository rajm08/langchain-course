from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

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


llm = ChatOpenAI(model="gpt-4.1-mini", 
                 temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs with AI ML skill set in Dallas, Texas")})
    print(result)
    #print(result.content)



if __name__ == '__main__' :
    main()
