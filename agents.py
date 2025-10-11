from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
import os

tools = [
    TavilySearch()
]

llm = ChatOllama(model="qwen3:0.6b", base_url="http://localhost:11434")

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, prompt=react_prompt, tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    handle_parsing_errors=True,
                                                    verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={"input": "search for 3 job postings for data scientist in san francisco and list their details"}
        )
    
    print(result)
   
if __name__ == "__main__":
    main()
