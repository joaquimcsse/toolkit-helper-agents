from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

import os
from typing import List, Any

from .tools import (list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, 
            evaluate_classification_dataset, evaluate_regression_dataset)
from project.utils import add, subtract, multiply, divide, power
from project.base_agent import Agent


class DataAgent(Agent):
    """DataAgent that extends the Agent class."""
    def __init__(self):
        self.tools = [list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, 
                evaluate_classification_dataset, evaluate_regression_dataset]
        self.restrictions = {"max_tokens": 200, "max_retries": 3}
        self.llm = None
        self.agent = None


    def _start_agent(self):

        self.llm = ChatOpenAI(
            model="gpt-5-nano", max_tokens=self.restrictions["max_tokens"], max_retries=self.restrictions["max_retries"]
            )

        prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "You are a data science assistant. Use the available tools to analyze CSV files. "),
            
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")  
        ])

        self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        agent_executor.agent.stream_runnable = False

        print("Ask questions about the datasets on the current directory (type 'exit' to quit):")

        while True:
            user_input=input("Request some data task:")
            if user_input.strip().lower() in ['exit','quit']:
                print("shutting down...")
                break
            
            result=agent_executor.invoke({"input":user_input})
            print(f"my Agent: {result['output']}")

    def call(self):
        self._set_openai_key()
        self._start_agent()
        print("successfully shut down")
        