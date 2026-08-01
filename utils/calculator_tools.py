from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
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
