import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# GitHub Configuration
GITHUB_USERNAME = "knightscode139"
TOKEN_GITHUB = os.getenv("TOKEN_GITHUB")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "github-repos"

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OPENAI Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
