from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import re
import sys
import pandas as pd

from project.base_agent import Agent

class DataAnalyst(Agent):

    def __init__(self):
        self._obtain_csv_name()
        try:
            self.df = pd.read_csv(self.csv_fp)
        except FileNotFoundError:
            sys.exit("csv file not found")
        
        self.restrictions = {"max_tokens": 200, "max_retries": 3}
        self.llm = None
        self.agent = None

    
    def _obtain_csv_name(self):
        """Get the name of the csv for analysis from command line input.
        """
        from re import search
        try:
            entered = str(input("enter csv file name (example.csv):"))
            if re.search(r"\.csv$", entered):
                self.csv_fp = entered
            else:
                sys.exit("pass a valid csv file name. exiting.")
        except Exception:
            sys.exit("error on: '_obtain_csv_name'")
    

    def _start_agent(self, credentials, model_id):

        params = {
            GenParams.MAX_NEW_TOKENS: 256, 
            GenParams.TEMPERATURE: 0
        }

        model = Model(
            model_id=model_id, 
            credentials=credentials, 
            params=params
        )

        self.llm = WatsonxLLM(model = model)

        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=False,
            return_intermediate_steps=True, 
            handle_parsing_errors=True,
            prefix="You are a pandas agent. Always respond using Action/Action Input/Final Answer format. Never output raw data directly."
        )

        while True:
            user_input=input("Write a query in natural language:")
            if user_input.strip().lower() in ['exit','quit']:
                print("shutting down...")
                break
            
            result=self.agent.invoke({"input":user_input})
            print(f"Response: {result['output']}")


    def call(self):
        credentials = self._obtain_credentials()
        model_id = self._get_model_id()
        self._start_agent(credentials, model_id)
        print("successfully shut down")
