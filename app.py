import chromadb
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import gradio as gr
import os

# Download pre-built database from HuggingFace Dataset
print("Downloading ChromaDB from HuggingFace...")
db_path = snapshot_download(
    repo_id="knightscode139/github-repos-chromadb",
    repo_type="dataset",
    local_dir="./chroma_db"
)
print(f"Database downloaded to: {db_path}")

# Load ChromaDB
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="github_repos")
total_docs = collection.count()
print(f"Collection loaded: {total_docs} documents")

# Get portfolio stats
all_data = collection.get()
repo_names = list(set([meta['repo'] for meta in all_data['metadatas']]))
repo_list = ", ".join(repo_names)
print(f"Repositories: {repo_list}")

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Claude client
print("Initializing Claude client...")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Setup complete. Ready to answer questions!")

def answer_question(question):
    """Main RAG pipeline: retrieve relevant code and generate answer"""
    
    try:
        # Step 1: Convert question to embedding
        print(f"Question: {question}")
        query_embedding = embedding_model.encode(question).tolist()
        
        # Step 2: Search ChromaDB for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Step 3: Extract retrieved documents
        retrieved_docs = results['documents'][0]
        context = "\n\n---\n\n".join(retrieved_docs)
        
        # Step 4: Query Claude
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            temperature=0.3,
            top_p=0.7,
            system=f"""You are knightscode139's portfolio assistant. Answer in FIRST PERSON as knightscode139.

Portfolio: {total_docs} files across repositories: {repo_list}

CRITICAL RULES:
1. ONLY state what is EXPLICITLY shown in the code context below
2. If something is NOT mentioned in the code, say "I don't see that specific detail in my code"
3. NEVER add standard practices or assumptions (like "I probably used X" or "typical approaches include Y")
4. Be CONCISE (2-4 sentences)
5. When asked about repos/projects, list ALL from portfolio info above
6. Decline off-topic questions

Code context:
{context}""",
            messages=[{"role": "user", "content": question}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error: {str(e)}. Please try again or contact the developer."

# Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask about my projects",
        placeholder="Example: How did you handle class imbalance in the COVID classifier?",
        lines=2
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="🤖 knightscode139's GitHub Portfolio Chatbot",
    description="Ask questions about my code and projects. Powered by RAG + Claude Haiku.",
    examples=[
        ["What projects do you have?"],
        ["How did you train the COVID-19 classifier?"],
        ["What accuracy did you achieve on IMDB sentiment analysis?"],
        ["Show me your data preprocessing approach."]
    ]
)

# Launch the app
demo.launch()
