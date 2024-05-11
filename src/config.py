import os
from dotenv import load_dotenv

# Load environment variables from the .env file into the application's environment.
load_dotenv()

class Config:
    # Class to hold configuration constants and environment variables.

    # Slack API token for interacting with Slack APIs.
    SLACK_TOKEN = os.getenv("SLACK_TOKEN")
    # Version of the Azure OpenAI API being used.
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    # Name of the deployed Azure OpenAI service.
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    # API key for Azure OpenAI services.
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    # Endpoint URL for the Azure OpenAI API.
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    # Deployment name for Azure OpenAI embedding service.
    AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    # UUID for the Calendly event type, used for scheduling links.
    CALENDLY_EVENT_UUID = os.getenv("CALENDLY_EVENT_UUID")
    # API key for Calendly integration.
    CALENDLY_API_KEY = os.getenv("CALENDLY_API_KEY")
    # API key for LangChain Smith services.
    LANGCHAIN_SMITH_API_KEY = os.getenv("LANGCHAIN_SMITH_API_KEY")
    # Fixed endpoint for LangChain Smith API services.
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    # Identifier for the project in LangChain services.
    LANGCHAIN_PROJECT = "EchoLink AI"
    # Enable detailed tracing for LangChain operations.
    LANGCHAIN_TRACING_V2 = "True"
