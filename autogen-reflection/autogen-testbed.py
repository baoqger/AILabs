import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"), # deployed model name
        model=os.getenv("AZURE_OPENAI_MODEL"), # the name of OpenAI model
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # For key-based authentication.
    )

    agent1 = AssistantAgent("assistant1", model_client=client, system_message="You are a writer, write well.")
    agent2 = AssistantAgent(
        "assistant2",
        model_client=client,
        system_message="You are an editor, provide critical feedback. Respond with 'APPROVE' if the text addresses all feedbacks.",
    )
    inner_termination = TextMentionTermination("APPROVE")
    inner_team = RoundRobinGroupChat([agent1, agent2], termination_condition=inner_termination)

    society_of_mind_agent = SocietyOfMindAgent("society_of_mind", team=inner_team, model_client=client)

    agent3 = AssistantAgent(
        "assistant3", model_client=client, system_message="Translate the text to Spanish."
    )
    team = RoundRobinGroupChat([society_of_mind_agent, agent3], max_turns=2)

    stream = team.run_stream(task="Write a short story with a surprising ending.")
    await Console(stream)


asyncio.run(main())
