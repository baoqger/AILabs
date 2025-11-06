from autogen_core.models import UserMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

load_dotenv()

async def main():

    az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"), # deployed model name
        model=os.getenv("AZURE_OPENAI_MODEL"), # the name of OpenAI model
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # For key-based authentication.
    )

    result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result.content)
    await az_model_client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())