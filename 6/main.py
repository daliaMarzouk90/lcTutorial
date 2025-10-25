from dotenv import load_dotenv
from langchain.tools import tool, Tool
from langchain.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_ollama import ChatOllama
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List

from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters"""
    print(f"get_text_length received input: {text}")
    text = text.strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

if __name__ == '__main__':
    print("Hello ReaAct LangChain!")
    tools = [get_text_length]
    
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([tool.name for tool in tools]))
    
    
    
    llm = ChatOllama(model="qwen2.5:3b",
                     base_url="http://localhost:11434", 
                      temperature=0, 
                     stop=["\nObservation"],
                     callbacks=[AgentCallbackHandler()]) #we add the stop to avoid the LLM to hallucinate after the observation

    intermediate_steps = []
    
    # format_log_to_str 

    agent = {"input": lambda x:x["input"],
             "agent_scratchpad": lambda x:x["agent_scratchpad"]
            } | prompt | llm | ReActSingleInputOutputParser()  # Placeholder for agent input handling
    # agent = {"input": lambda x:x["input"]} | prompt | llm  # Placeholder for agent input handling

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        input={
            "input": 'What is the length in characters of the text Dalia ?',
            "agent_scratchpad": intermediate_steps,
            }
    )
    
    print(agent_step)
    counter = 0
    
    while not isinstance(agent_step, AgentFinish):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        
        observation = tool_to_use.func(str(tool_input))
        print("-----")
        print(agent_step)
        print(f"Observation#{counter}: {observation}")
        intermediate_steps.append((agent_step, str(observation)))
        
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            input={
                "input": 'What is the length in characters of the text Dalia ?',
                "agent_scratchpad": intermediate_steps,
                }
        )
        
        counter += 1
    print(agent_step.return_values)