# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
import os
from dotenv import load_dotenv
from langchain.tools import tool

# Load environment variables from .env
load_dotenv()

endpoint= os.getenv("AZURE_INFERENCE_ENDPOINT")
credential = os.getenv("AZURE_INFERENCE_CREDENTIAL")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")


model = AzureAIChatCompletionsModel(
    endpoint=endpoint,
    credential=credential,
    model=model_deployment,
)



@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(result)