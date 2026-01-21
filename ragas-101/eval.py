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
            '张伟是哪个部门的？',
            '张伟是哪个部门的？',
            '张伟是哪个部门的？'
        ],
        'answer': [
            '根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。',
            '张伟是人事部门的',
            '张伟是教研部的'
        ],
        'ground_truth':[
            '张伟是教研部的成员',
            '张伟是教研部的成员',
            '张伟是教研部的成员'
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