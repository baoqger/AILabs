# 导入依赖
from urllib import response
from urllib import response
from llama_index.core import Settings, SimpleDirectoryReader,VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
# 这两行代码是用于消除 WARNING 警告信息，避免干扰阅读学习，生产环境中建议根据需要来设置日志级别
import logging
from llama_index.core.node_parser import SemanticSplitterNodeParser
from networkx import nodes
logging.basicConfig(level=logging.ERROR)
import os
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import shutil
load_dotenv()

llm = AzureOpenAI(
    engine=os.getenv("AZURE_OPENAI_MODEL"),
    model=os.getenv("AZURE_OPENAI_MODEL"), 
    temperature=0.0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    engine=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
)

res = llm.complete("The sky is a beautiful blue and")
print(res)