from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper(serpapi_api_key="1d4e507de74dae6e79f9a0373952a620847f99c2e47d4bd57524fdf4481da875")

search_tool = Tool(
    name="SearchWeb",
    func=search.run,
    description="Recherche des informations actuelles sur le web via SerpAPI"
)
from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama

# Ton LLM
llm = ChatOllama(model="llama3.1", temperature=0)

# L’agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)
agent.run("Qui est le roi du Maroc ?")
