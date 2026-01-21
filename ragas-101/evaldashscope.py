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


    dataset = Dataset.from_dict(data_samples)
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness],
        llm=Tongyi(model_name="qwen-plus"),
        embeddings=DashScopeEmbeddings(model="text-embedding-v3")
    )
    return score.to_pandas()