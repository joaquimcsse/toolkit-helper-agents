import json
import os
from typing import List, Any

from .utils import make_chain

class Agent():
    """Base Agent class providing some common functionality for concrete agents."""
    
    def _obtain_credentials(self):
        import getpass
        url = getpass.getpass("Enter LLM url (ex. 'https://us-south.ml.cloud.ibm.com'):").strip()
        key = getpass.getpass("Enter API key:").strip()
        if not url or not key:
            print("No API key provided. Exiting.")
            raise SystemExit(1)
        return {
                "url": url,
                "api_key":key
            }
    

    def _set_openai_key(self):
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

    
    def _get_model_id(self):
        id = str(input("Enter model id (ex. 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'):")).strip()
        if not id:
            print("No id. Exiting.")
            raise SystemExit(1)
        return id


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


    def _chain(self, query : dict, tool_mapping : dict, llm):
        make_chain()
        