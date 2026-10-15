if __name__ == "__main__":
    from toolkit_helper_agents.nl_tasks.data_analyst import DataAnalyst
    from toolkit_helper_agents.df_tasks.data_agent import DataAgent

    #agent = DataAgent()
    agent = DataAnalyst()
    agent.call()
    