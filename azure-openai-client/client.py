import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
model_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    model=deployment
)

print(response.choices[0].message.content)