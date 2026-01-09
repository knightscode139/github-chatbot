import requests
import base64
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from src.config import *


TEXT_FILES = ['.py', '.ipynb', '.md', '.txt', '.yml', '.json', '.sh', '.yml']


def get_splitter_for_extension(ext: str):
    """Returns appropriate text splitter based on file extension."""
    if ext == '.py':
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
    elif ext == '.md':
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
        )
    elif ext == '.ipynb':
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1500, chunk_overlap=300
        )
    else:
        # Generic splitter for .txt, .yml, .yaml, .json, .sh
        return RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )


def fetch_all_repos():
    """Fetches all public repositories for the configured GitHub user."""
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    headers = {'Authorization': f'token {TOKEN_GITHUB}'} if TOKEN_GITHUB else {}
    response = requests.get(url, headers=headers)
    repos = response.json()
    
    # Check if we got an error
    if isinstance(repos, dict) and 'message' in repos:
        print(f"ERROR: {repos['message']}")
        return []
    
    # Filter out forked repositories
    return [repo['name'] for repo in repos if not repo.get('fork', False)]


def fetch_repo_files(repo_name):
    """Fetches all text files from a repository."""
    headers = {'Authorization': f'token {TOKEN_GITHUB}'} if TOKEN_GITHUB else {}
    tree_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/trees/main?recursive=1"
    response = requests.get(tree_url, headers=headers)
    
    tree = response.json()
    files = []
    for item in tree.get('tree', []):
        if item['type'] == 'blob' and any(item['path'].endswith(ext) for ext in TEXT_FILES):
            files.append(item)
    return files


def download_file_content(repo_name, file_path):
    """Downloads the content of a specific file from GitHub."""
    headers = {'Authorization': f'token {TOKEN_GITHUB}'} if TOKEN_GITHUB else {}
    content_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    response = requests.get(content_url, headers=headers)
    data = response.json()
    content = base64.b64decode(data['content']).decode('utf-8')
    return content


def fetch_all_content():
    """Fetches all text files from all repositories."""
    all_repos = fetch_all_repos()
    print(f"Found {len(all_repos)} repositories")
    
    all_content = []
    for repo_name in all_repos:
        print(f"\n--- Processing {repo_name} ---")
        try:
            files = fetch_repo_files(repo_name)
            print(f"Found {len(files)} text files")
            for file in files:
                try:
                    content = download_file_content(repo_name, file['path'])
                    all_content.append({
                        'repo': repo_name,
                        'path': file['path'],
                        'content': content
                    })
                    print(f"  ✓ {file['path']}")
                except Exception as e:
                    print(f"  ✗ Failed: {file['path']} - {e}")
        except Exception as e:
            print(f"Failed to process repo {repo_name}: {e}")

    return all_content


def build_pinecone_index(data):
    """Builds embeddings and uploads to Pinecone."""
    print("\n=== Building Pinecone Index ===")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")


    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Connect to the index
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to index. Current vector count: {index.describe_index_stats()['total_vector_count']}")
    # Process each document
    print(f"\nProcessing {len(data)} files...")
    vectors_to_upsert = []

    for idx, item in enumerate(data):
        # Get file extension to determine splitter
        file_ext = '.' + item['path'].split('.')[-1]
        splitter = get_splitter_for_extension(file_ext)
        # Create LangChain Document
        doc = Document(
            page_content=item['content'],
            metadata={'repo': item['repo'], 'path': item['path']}
        )
        # Split document into chunks
        chunks = splitter.split_documents([doc])
        print(f"  [{idx+1}/{len(data)}] {item['repo']}/{item['path']} → {len(chunks)} chunks")
        
        # Generate embeddings for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            # Create embedding
            embedding = model.encode(chunk.page_content).tolist()
            # Create unique ID
            vector_id = f"{item['repo']}_{item['path']}_{chunk_idx}"

            # Prepare vector for Pinecone
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'repo': item['repo'],
                    'path': item['path'],
                    'chunk': chunk_idx,
                    'text': chunk.page_content[:1000]  # Store first 1000 chars
                }
            })
    
    # Upload to Pinecone in batches
    print(f"\nUploading {len(vectors_to_upsert)} vectors to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Uploaded batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")

    print(f"\n✅ Done! Total vectors in index: {index.describe_index_stats()['total_vector_count']}")
    

if __name__ == "__main__":
    print("=== GitHub Portfolio RAG - Indexing Pipeline ===\n")

    # Step 1: Fetch all content from GitHub
    print("STEP 1: Fetching GitHub repositories...")
    data = fetch_all_content()
    print(f"\nTotal files fetched: {len(data)}")

    # Step 2: Build Pinecone index
    print("\nSTEP 2: Building embeddings and uploading to Pinecone...")
    build_pinecone_index(data)

    print("\n" + "="*50)
    print("✅ Indexing complete!")
    print(f"Your Pinecone index '{PINECONE_INDEX_NAME}' is ready.")
    print("="*50)
