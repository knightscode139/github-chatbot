# GitHub Portfolio RAG Chatbot

An AI-powered chatbot that answers questions about my code and projects using Retrieval-Augmented Generation (RAG).

## 🔗 Live Demo

[Try it here](https://huggingface.co/spaces/knightscode139/github-portfolio-chatbot)

## ✨ Features

- Automatically indexes all my GitHub repositories
- Answers questions about code, architecture, and implementation details
- Powered by ChromaDB vector database + Claude Haiku
- Updates via command-line script when new repos are added

## 🛠️ Tech Stack

- **Vector Database:** ChromaDB
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM:** Claude 3.5 Haiku
- **Frontend:** Gradio
- **Deployment:** HuggingFace Spaces

## 🧠 How It Works

1. **Indexing:** Fetches all text files from GitHub repos, generates embeddings, stores in ChromaDB
2. **Retrieval:** User question → embedding → finds 3 most similar code chunks
3. **Generation:** Claude reads retrieved chunks and generates answer in first person

## 📂 Project Structure
```
├── build_database.py       # Fetches repos, builds ChromaDB, uploads to HF
├── app.py                  # Gradio Space application
├── requirements.txt        # Dependencies
└── README.md
