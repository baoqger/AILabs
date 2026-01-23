from tqdm.cli import tqdm as tqdm_cli
import tqdm.auto
tqdm.auto.tqdm = tqdm_cli

from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from dotenv import load_dotenv
import dashscope
import os
load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def eval_dashscope():
    data_samples = {
        'question': [
            "What is depth match? Please directly answer the question do not show any reference",
        ],
        'answer': [
            "Depth matching is the process of aligning two well logs by adjusting the depths of one log (the target) so that its measured features correspond to those of the other log (the reference). "
,
        ],
        'ground_truth':[
            "Depth matching is the process of aligning two well logs by adjusting the depths of one log (the target) so that its measured features correspond to those of the other log (the reference). "
,
        ]
    }


    dataset = Dataset.from_dict(data_samples)
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness],
        llm=Tongyi(model_name="qwen-plus"),
        embeddings=DashScopeEmbeddings(model="text-embedding-v3")
    )
    return score.to_pandas()