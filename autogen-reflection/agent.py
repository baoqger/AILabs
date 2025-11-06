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

    task = '''
            Write a concise but engaging blogpost about
        DeepLearning.AI. Make sure the blogpost is
        within 100 words.
        '''

    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"), # deployed model name
        model=os.getenv("AZURE_OPENAI_MODEL"), # the name of OpenAI model
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # For key-based authentication.
    )

    writer = AssistantAgent(
        name="Writer",
        system_message="You are a writer. You write engaging and concise " 
            "blogpost (with title) on given topics. You must polish your "
            "writing based on the feedback you receive and give a refined "
            "version. Only return your final work without additional comments.",
        model_client=client,
    )

    # response = await writer.on_messages(
    #     [TextMessage(content=task, source="user")],
    #     cancellation_token=CancellationToken(),
    # )   
    # print("Agent response:", response.chat_message.content) 

    # critic agent
    critic = AssistantAgent(
        name="Critic",
        #is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        model_client=client,
        system_message="You are a critic. You review the work of "
                    "the writer and provide constructive "
                    "feedback to help improve the quality of the content.",
    )  

    res = critic.initiate_chat(
        recipient=writer,
        message=task,
        max_turns=2,
        summary_method="last_msg"
    )  

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())