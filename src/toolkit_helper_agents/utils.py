from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool

def make_chain():
        generic_chain = (
            RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
            | RunnableLambda(lambda messages: messages+[llm.invoke(messages)])
            | RunnableLambda(lambda messages: _recursive_chain(messages, llm, tool_mapping))
        )
        return generic_chain

def _execute_tool(tool_call, tool_mapping):
    """Execute single tool call and return ToolMessage""" 
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        content= f"Error: {str(e)}"
        
    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"]
    )


def _process_tool_calls(messages, llm, tool_mapping):
    """Recursive tool call processor"""
    last_message = messages[-1]

    tool_messages = [
        _execute_tool(tc, tool_mapping)
        for tc in getattr(last_message, 'tool_calls', [])
    ]

    # Add tool responses to message history
    updated_messages = messages + tool_messages
        
    # Get next LLM response
    next_ai_response = llm.invoke(updated_messages)

    return updated_messages + [next_ai_response]


def _should_continue(messages):
    """Check if another iteration is needed"""
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))


def _recursive_chain(messages, llm, tool_mapping):
    """Recursively process tool calls until completion"""
    if _should_continue(messages):
        new_messages = _process_tool_calls(messages, llm, tool_mapping)
        return _recursive_chain(new_messages, llm, tool_mapping)
    return messages

@tool
def add(self, a: int, b: int) -> int:
    '''Add a (int) and b (int).'''
    return a+b


@tool
def subtract(a: int, b: int) -> int:
    '''Subtract b (int) from a (int).'''
    return a-b


@tool
def multiply(a: int, b: int) -> int:
    '''Multiply a and b.'''
    return a*b


@tool 
def divide(a: int, b: int) -> float:
    '''Divide a (int) by b (int).'''
    return a/b


@tool
def power(a: int, b: int) -> int:
    '''Raises a to the power of b'''
    return a**b
