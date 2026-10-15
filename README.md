# LLM Based Autonomous Agents With LangChain

Setting up LLM based autonomous agents with LangChain's wrappers and tools.

## Requirements 
Download the required extensions (using uv or your preferred package manager)
```bash
    uv sync
```

## Usage 
```python
    if __name__ == "__main__":
        from toolkit_helper_agents.nl_tasks.data_analyst import DataAnalyst
        from toolkit_helper_agents.df_tasks.data_agent import DataAgent

        #agent = DataAgent()
        #agent = DataAnalyst()
        agent.call()
```
