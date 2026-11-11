from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


@tool
def get_weather(city: str) -> str:
    """
    returns the weather in the given city

    Args:
        city: str name of the city
    
    returns:
        str: weather in the given city
    """

    return f"it is very sunny in {city}"


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[get_weather]
)

res = agent.invoke({"messages":HumanMessage(content="What is the weather right now in hyderabad")})

print(res["messages"][-1].content)