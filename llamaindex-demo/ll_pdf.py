import openai
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from dotenv import load_dotenv
import pypdf
from llama_index.core import Settings

load_dotenv()

llm = AzureOpenAI(
    engine=os.getenv("AZURE_OPENAI_MODEL"),
    model=os.getenv("AZURE_OPENAI_MODEL"), 
    temperature=0.0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

#response = llm.complete("The sky is a beautiful blue and")
# print(response)
embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    engine=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
)

# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
# )
# set_global_service_context(service_context)

Settings.llm = llm
Settings.embed_model = embed_model

input_file_path = 'attention_is_all_you-need.pdf'

documents = SimpleDirectoryReader(
    input_files=[input_file_path]
).load_data()
index = VectorStoreIndex.from_documents(documents)

query = "explain Scaled Dot-Product Attention in short and bullets"
query_engine = index.as_query_engine()
answer = query_engine.query(query)
print(answer.response)

