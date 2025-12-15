import requests
import base64
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import HfApi
import os

username = "knightscode139"
TEXT_FILES = ['.py', '.ipynb', '.md', '.txt', '.yaml', '.json', '.sh']

# ===== STEP 1: FETCH REPOS =====

def fetch_all_repos():
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    repos = response.json()
    return [repo['name'] for repo in repos]

def fetch_repo_files(repo_name):
    tree_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/main?recursive=1"
    response = requests.get(tree_url)
    tree = response.json()
    
    files = []
    for item in tree.get('tree', []):
        if item['type'] == 'blob' and any(item['path'].endswith(ext) for ext in TEXT_FILES):
            files.append(item)
    return files

def download_file_content(repo_name, file_path):
    content_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}"
    response = requests.get(content_url)
    data = response.json()
    content = base64.b64decode(data['content']).decode('utf-8')
    return content

def fetch_all_content():
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

# ===== STEP 2: BUILD CHROMADB =====

def build_chromadb(data):
    print("\n=== Building ChromaDB ===")
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Creating ChromaDB client...")
    # Remove old database if exists
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.create_collection(name="github_repos")
    
    print(f"Processing {len(data)} files...")
    for idx, item in enumerate(data):
        # Create document text
        doc_text = f"Repository: {item['repo']}\nFile: {item['path']}\n\n{item['content']}"
        
        # Generate embedding
        embedding = model.encode(doc_text).tolist()
        
        # Store in ChromaDB
        collection.add(
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{"repo": item['repo'], "path": item['path']}],
            ids=[f"{item['repo']}_{idx}"]
        )
        print(f"  ✓ {idx+1}/{len(data)}: {item['repo']}/{item['path']}")
    
    print("ChromaDB built successfully!")

# ===== STEP 3: UPLOAD TO HUGGINGFACE =====

def upload_to_hub(hf_token):
    print("\n=== Uploading to HuggingFace ===")
    api = HfApi()
    
    # Create dataset repo if it doesn't exist
    try:
        api.create_repo(
            repo_id="knightscode139/github-repos-chromadb",
            repo_type="dataset",
            token=hf_token,
            exist_ok=True
        )
    except:
        pass  # Repo already exists
    
    # Upload folder
    api.upload_folder(
        folder_path="./chroma_db",
        repo_id="knightscode139/github-repos-chromadb",
        repo_type="dataset",
        token=hf_token
    )
    print("Upload complete!")

# ===== MAIN =====

if __name__ == "__main__":
    print("=== STEP 1: Fetching GitHub Repos ===")
    data = fetch_all_content()
    print(f"\nTotal files fetched: {len(data)}")
    
    print("\n=== STEP 2: Building Embeddings & ChromaDB ===")
    build_chromadb(data)
    
    print("\n=== STEP 3: Uploading to HuggingFace ===")
    hf_token = input("Enter your HuggingFace token (or set HF_TOKEN env variable): ").strip()
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: No HuggingFace token provided")
        exit(1)
    
    upload_to_hub(hf_token)
    
    print("\n✅ Done! Database uploaded to: huggingface.co/datasets/knightscode139/github-repos-chromadb")
