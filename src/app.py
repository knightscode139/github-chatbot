from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import gradio as gr
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from src.config import *

# Initialize Pinecone
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Get portfolio stats
stats = index.describe_index_stats()
total_vectors = stats['total_vector_count']
print(f"Connected! Index contains {total_vectors} vectors")


@tool(response_format="content_and_artifact")
def retrieve_code_context(query: str):
    """Search through my GitHub repositories to find relevant code and project information."""
    # Convert query to embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Format results for the LLM
    context_parts = []
    for match in results['matches']:
        repo = match['metadata']['repo']
        path = match['metadata']['path']
        text = match['metadata']['text']
        score = match['score']
        context_parts.append(f"[Repo: {repo}, File: {path}, Relevance: {score:.2f}]\n{text}")

    serialized = "\n\n---\n\n".join(context_parts)
    return serialized, results['matches']


# Initialize Claude for LangChain
print("Initializing Claude agent...")
model = init_chat_model(
    "claude-sonnet-4-20250514",
    model_provider="anthropic",
    api_key=ANTHROPIC_API_KEY
)

# Create RAG agent with retrieval tool
tools = [retrieve_code_context]
system_prompt = f"""You are knightscode139's AI portfolio assistant. You have access to a tool that searches through {total_vectors} code chunks from his GitHub repositories.

CRITICAL RULES:
1. Use the search tool to find relevant code before answering technical questions
2. Answer in FIRST PERSON as knightscode139
3. ONLY state what is EXPLICITLY shown in the retrieved code
4. If code doesn't contain specific details, say "I don't see that in my code"
5. Be CONCISE (2-4 sentences unless asked for more detail)
6. Decline off-topic questions politely

When you retrieve code, cite the repo and file name naturally in your response."""

agent = create_agent(model, tools, system_prompt=system_prompt)
print("Agent ready!")


def answer_question(question, history):
    """Handle user questions with the RAG agent."""

    try:
        # Convert Gradio history to LangChain messages
        messages = []
        for msg in history:
            messages.append({
                "role": msg['role'],
                "content": msg['content'][0]['text']  # Extract text from nested structure
            })
            
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Stream agent responses
        response_text = ""
        for event in agent.stream(
            {"messages": messages},
            stream_mode="values"
        ):
            last_message = event["messages"][-1]
            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                response_text = last_message.content
        
        return response_text
        
    except Exception as e:
        return f"Error: {str(e)}. Please try again."


# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=answer_question,
    title="🤖 knightscode139's GitHub Portfolio Assistant",
    description=f"""Ask questions about my code and projects! Powered by LangChain RAG Agent + Claude Sonnet 4.
    
**Currently indexed:** {total_vectors} code chunks from my GitHub repositories.
    
The agent can search through my code multiple times to give you accurate answers.""",
    examples=[
        "What projects do you have?",
        "How did you handle data preprocessing?",
        "Show me your experience with machine learning",
        "What accuracy did you achieve in your models?",
        "Do you have any NLP projects?"
    ],
)


if __name__ == "__main__":
    demo.launch()
