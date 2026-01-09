# GitHub Portfolio RAG Chatbot

AI-powered chatbot that answers questions about my code using Retrieval-Augmented Generation (RAG) with LangChain agents.

## 🔗 Live Demo

[Try it here](https://huggingface.co/spaces/knightscode139/github-portfolio-chatbot)

## ✨ Features

- **Intelligent Retrieval**: LangChain agent decides when to search, can search multiple times
- **Language-Aware Chunking**: Python, Markdown, and code-specific text splitters
- **Auto CI/CD**: Daily automated index updates via GitHub Actions
- **Modern Stack**: Pinecone vector database + Claude Sonnet 4.5
- **Auto-indexing**: Fetches all GitHub repos, filters out forks

## 🛠️ Tech Stack

- **Vector Database:** Pinecone (serverless)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM:** Claude Sonnet 4.5
- **Framework:** LangChain
- **Frontend:** Gradio ChatInterface
- **Deployment:** HuggingFace Spaces
- **CI/CD:** GitHub Actions

## 🧠 How It Works

1. **Indexing:** Fetches repos via GitHub API → splits with language-aware splitters → embeds → uploads to Pinecone
2. **Retrieval:** Agent converts question to embedding → searches Pinecone → can search again if needed
3. **Generation:** Claude reads retrieved chunks and answers in first person

## 📂 Project Structure
```
├── .github/
│   └── workflows/
│       └── update_index.yml   # Daily CI/CD pipeline
├── src/
│   ├── indexing/
│   │   └── build_index.py     # GitHub fetching + Pinecone indexing
│   ├── app.py                 # Gradio ChatInterface logic
│   └── config.py              # API keys and settings
├── app.py                     # HF Spaces entry point
├── requirements.txt           # HF deployment dependencies
├── pyproject.toml             # UV dependencies
└── README.md
```

## 🚀 Setup
```bash
# Install dependencies
uv sync

# Set environment variables
export PINECONE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export _GITHUB_TOKEN="..."

# Build index
uv run python -m src.indexing.build_index

# Run app
uv run python src/app.py
```

## 🔄 CI/CD

GitHub Actions automatically updates the Pinecone index daily at 8 AM UTC:
- Fetches latest repos
- Re-indexes changed files
- Keeps chatbot in sync with GitHub

Configure secrets in GitHub repo settings: `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`, `TOKEN_GITHUB`
