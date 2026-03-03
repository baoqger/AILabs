import os

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
import json

from dotenv import load_dotenv

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

azure_credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
client = SearchIndexClient(endpoint = AZURE_SEARCH_ENDPOINT, credential = azure_credential)

# List existing indexes
indexes = client.list_indexes()

for index in indexes:
   index_dict = index.as_dict()
   print("Index Name:", index_dict['name'])