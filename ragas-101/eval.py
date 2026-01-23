from tqdm.cli import tqdm as tqdm_cli
import tqdm.auto
tqdm.auto.tqdm = tqdm_cli

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from dotenv import load_dotenv
import os
load_dotenv()

def eval_azure_openai():
    data_samples = {
        'question': [
            "What is depth match? Please directly answer the question do not show any reference",
        ],
        'answer': [
            "Depth matching is the process of aligning two well logs by adjusting the depths of one log (the target) so that its measured features correspond to those of the other log (the reference). ",
        ],
        'ground_truth':[
            "Depth matching is the process of aligning two well logs by adjusting the depths of one log (the target) so that its measured features correspond to those of the other log (the reference). ",
        ]
    }

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),  # or your deployment
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")    
    )

    dataset = Dataset.from_dict(data_samples)
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=embeddings
    )
    return score.to_pandas()


def evaluate_result(question, response, ground_truth):
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),  # or your deployment
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")    
    )
    # 获取回答内容
    if hasattr(response, 'response_txt'):
        answer = response.response_txt
    else:
        answer = str(response)

    data_samples = {
        'question': [question],
        'answer': [answer],
        'ground_truth':[ground_truth],
        # 'contexts' : [context],
    }
    dataset = Dataset.from_dict(data_samples)                
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=embeddings
    )
    return score.to_pandas()