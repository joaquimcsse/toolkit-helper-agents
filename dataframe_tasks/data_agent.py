from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

import os
from typing import List, Any
from .tools import (list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, 
            evaluate_classification_dataset, evaluate_regression_dataset)
from utils.calculator_tools import add, subtract, multiply, divide, power


class Agent():
    """Base Agent class providing common functionality for concrete agents."""
    
    def _obtain_api_key(self):
        """Get api key from line command input
        Prompts securely (hidden input) when needed.
        """
        import getpass
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return key

        entered = getpass.getpass("Enter OpenAI API key: ").strip()
        if not entered:
            print("No API key provided. Exiting.")
            raise SystemExit(1)

        os.environ["OPENAI_API_KEY"] = entered
        return entered


    def find_first_tool(self, response):
        action = response[0]

        print("the agent decided to call a tool:")
        print("Tool Name:", action.tool)
        print("Tool Input:", action.tool_input)
        print("Log:\n", action.log.strip())


    def add_tool(self, tools: List[Any]):
        """Bind additional tools to the underlying LLM if initialized."""
        if not self.llm:
            self.tools.extend(tools)
            return
        try:
            self.llm = self.llm.bind_tools(tools)
        except Exception:
            self.tools.extend(tools)


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
        self.add_tool([add, subtract, multiply, divide, power])

        prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a data science assistant. Use the available tools to analyze CSV files. "),
        
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")  
        ])

        self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        intro = self.agent.invoke({
        "input": "Introduce yourself",
        "intermediate_steps": []
        })

        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        agent_executor.agent.stream_runnable = False

        print("Ask questions about the datasets on the current directory (type 'exit' to quit):")

        while True:
            user_input=input(" You:")
            if user_input.strip().lower() in ['exit','quit']:
                print("shutting down...")
                break
            
            result=agent_executor.invoke({"input":user_input})
            print(f"my Agent: {result['output']}")

    def call(self):
        self._obtain_api_key()
        self._start_agent()
        print("successfully shut down")
        