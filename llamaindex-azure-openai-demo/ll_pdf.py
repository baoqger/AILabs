import openai
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from dotenv import load_dotenv
import pypdf
from llama_index.core import Settings, StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

llm = AzureOpenAI(
    engine=os.getenv("AZURE_OPENAI_MODEL"),
    model=os.getenv("AZURE_OPENAI_MODEL"), 
    temperature=0.0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #api_version=os.getenv("OPENAI_API_VERSION"),
)

embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    engine=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
)

Settings.llm = llm
Settings.embed_model = embed_model

# Create a Chroma client and collection
#chroma_client = chromadb.EphemeralClient()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.create_collection("ollama")
# Set up the ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Set up the StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

input_file_path = 'attention_is_all_you-need.pdf'

documents = SimpleDirectoryReader(
    input_files=[input_file_path]
).load_data()

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

query = "explain Scaled Dot-Product Attention in short and bullets"
query_engine = index.as_query_engine()
answer = query_engine.query(query)
print(answer.response)

