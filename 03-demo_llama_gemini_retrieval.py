"""
pip install llama-cloud-services
pip install llama-index-llms-google-genai
pip install llama-index-embeddings-google-genai

"""

"""
pip install llama-cloud-services
pip install llama-index-llms-google-genai
pip install llama-index-embeddings-google-genai
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings

# Google GenAI Imports
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Modern LlamaCloud Import
from llama_cloud_services import LlamaCloudIndex

# 1. Load environment variables
load_dotenv()
llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
org_id = os.getenv("ORGANIZATION_ID")
google_api_key = os.getenv("GEMINI_API_KEY")

if not llama_cloud_key or not org_id or not google_api_key:
    raise ValueError("Missing required environment variables.")

# 2. Configure LlamaIndex to use Google GenAI
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=google_api_key
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/text-embedding-004",
    api_key=google_api_key
)

# 3. Connect to LlamaCloud Index
index = LlamaCloudIndex(
    name="AI_Proposal",
    project_name="Default",
    api_key=llama_cloud_key,
    organization_id=org_id
)

# 4. Create query engine directly from the index
query_engine = index.as_query_engine()

# 5. Run the query
query = "What is the capital of egypt?"
response = query_engine.query(query)

# 6. Print the response
print("Query response:")
print(response)
