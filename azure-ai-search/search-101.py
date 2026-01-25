from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
import json

service_endpoint = "https://new-search-8161.search.windows.net"
credential = DefaultAzureCredential()
client = SearchIndexClient(endpoint = service_endpoint, credential = credential)

# List existing indexe
indexes = client.list_indexes()

for index in indexes:
   index_dict = index.as_dict()
   print("Index Name:", index_dict['name'])
   print(json.dumps(index_dict, indent = 2))