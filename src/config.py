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

# Anthropic Configuration (kept for backward compatibility; not used if using OpenRouter/Deepseek)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OPENROUTER / Deepseek Configuration
# OpenRouter API key (used as OpenAI-compatible endpoint)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Optional headers OpenRouter may require for public keys
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER") or ""
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE") or ""
# Model name exposed by OpenRouter that routes to Deepseek assistant
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME") or "deepseek/assistant"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
