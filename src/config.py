"""
Configuration file for loading environment variables.
"""
import os
from dotenv import load_dotenv

class Config:
    """
    Loads environment variables from a .env file.
    """
    def __init__(self):
        load_dotenv()
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.DOCS_PATH = os.getenv("DOCS_PATH", "./documents")
        self.OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()