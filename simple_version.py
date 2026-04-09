from typing_extensions import TypedDict
from typing import Annotated
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Define context schema
class Context(TypedDict):
    user_id: str

# Tool: Get weather for a city
@tool
def get_weather(city: str) -> str:
    """Return weather information for a given city"""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    data = response.json()

    # Simplify the response
    current = data['current_condition'][0]
    temp = current['temp_C']
    desc = current['weatherDesc'][0]['value']

    return f"Weather in {city}: {desc}, Temperature: {temp}°C"

# Tool: Locate user based on context
@tool
def locate_user(context: Annotated[Context, "Injected context"]) -> str:
    """Look up the current user's city based on their user ID from context"""
    user_id = context.get('user_id', '')
    match user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'

# Create the LLM
llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0.3)

# Create checkpointer for conversation memory
checkpointer = InMemorySaver()

# Create the tools list
tools = [get_weather, locate_user]

# Create the agent (no prompt parameter - simpler version)
agent = create_react_agent(
    llm,
    tools,
    checkpointer=checkpointer
)

# Setup context
user_context = Context(user_id='ABC123')

config = {
    "configurable": {
        "thread_id": "my-conversation",
    },
    "context": user_context
}

# Example 1: Simple invoke (RECOMMENDED FOR LEARNING)
print("=== Example 1: Simple Invoke ===")
response = agent.invoke(
    {"messages": [("user", "Where am I and what's the weather?")]},
    config=config
)

# Print the final response
print(response['messages'][-1].content)
print("\n")

# Example 2: Continue the conversation (uses checkpointer memory)
print("=== Example 2: Follow-up Question ===")
response2 = agent.invoke(
    {"messages": [("user", "Will I need an umbrella?")]},
    config=config  # Same thread_id, so it remembers previous context
)

print(response2['messages'][-1].content)
print("\n")

# Example 3: Streaming version (ADVANCED)
print("=== Example 3: Streaming (Advanced) ===")
for event in agent.stream(
    {"messages": [("user", "Tell me a weather joke!")]},
    config=config,
    stream_mode="values"
):
    if "messages" in event:
        last_message = event["messages"][-1]
        if hasattr(last_message, 'content') and last_message.content:
            print(last_message.content)
            print("---")  # Separator to see streaming chunks
