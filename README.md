# Toolkit for Simple Agents

Testing LangChain's wrappers and tools for building agents.

## Requirements 
Download the required extensions in uv.lock with uv package manager
```bash
    uv sync
```

## Usage 
```python
    if __name__ == "__main__":
        from project.nl_tasks.data_analyst import DataAnalyst
        from project.df_tasks.data_agent import DataAgent

        #agent = DataAgent()
        #agent = DataAnalyst()
        agent.call()
```
