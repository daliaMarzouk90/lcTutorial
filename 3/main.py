from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

import os

tools = [
    TavilySearch()
]

llm = ChatOllama(model="qwen3:0.6b", base_url="http://localhost:11434", reasoning=True)
structured_llm = llm.with_structured_output(AgentResponse)

react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")


agent = create_react_agent(llm=llm, prompt=react_prompt_with_format_instructions, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools,handle_parsing_errors=True, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | parse_output

def main():
    result = chain.invoke(
        input={"input": "search for 3 job postings for data scientist in san francisco and list their details"}
        )
    
    print(result)
   
if __name__ == "__main__":
    main()
