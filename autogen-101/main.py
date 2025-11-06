from autogen_core.models import UserMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
import os
from dotenv import load_dotenv

load_dotenv()

async def main():

    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"), # deployed model name
        model=os.getenv("AZURE_OPENAI_MODEL"), # the name of OpenAI model
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # For key-based authentication.
    )

    #result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    # print(result.content)
    # await az_model_client.close()

    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=[],
        system_message="You are a travel agent that plans great vacations",
    )
    user_query = "Plan me a great sunny vacation"
    response = await agent.on_messages(
        [TextMessage(content=user_query, source="user")],
        cancellation_token=CancellationToken(),
    )   
    print("Agent response:", response.chat_message.content) 



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())