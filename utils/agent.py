from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

agent_tools = [TavilySearchResults(max_results=3)]

agent_llm = ChatGroq(model="llama3-70b-8192", temperature=0)


def create_agent(llm, tools: list):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are given a list of company names, find the stock symbol(s) for them.
                   You have access to the search tool tavily_search_results_json
                   You can use the tool if you do not know the symbol, if you know then there is no need to search
                   input: list of company names 
                   output: Return only the symbols separated by spaces. Don't add any type of punctuantion.
                """,
            ),
            MessagesPlaceholder(variable_name="input"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


agent_chain = create_agent(agent_llm, agent_tools)
