from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS as template
from langchain.tools import tool, Tool
from langchain_core.tools.render import render_text_description
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from typing import Union, List
from langchain.schema import AgentAction, AgentFinish

@tool
def get_text_length(text: str) -> int:
    """"Returns the length of a text in characters."""
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


tools = [get_text_length]

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names=[tool.name for tool in tools],
)

llm = ChatOllama(model="qwen3:0.6b", base_url="http://localhost:11434", reasoning=True, temperature=0, stop=["\nObservation:"])

# agent =(
#     {"input": lambda x: x["input"] | prompt | llm | ReActSingleInputOutputParser()}  # Placeholder for agent input handling
# )
agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()  # Placeholder for agent input handling

agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    input={"inpu t": "What is the length of the text 'Hello, world!'?"}
)

if isinstance(agent_step, AgentFinish):
    tool_name = agent_step.tool
    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = agent_step.tool_input
    
    observation = tool_to_use.func(str(tool_input))
    print(f"Observation: {observation}")