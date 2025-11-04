import chromadb
from chromadb.utils import embedding_functions
import os

# The core logic will still use the LLM for refinement.

# Initialize ChromaDB client
client = chromadb.CloudClient(
  api_key='ck-5TQfccVxt5e27hR2tgmvbqNg9DZmSRB66BCpM8FUPk87',
  tenant='002537c7-769f-4856-81e2-14fcb5f34971',
  database='dev'
)

# Use the local embedding function
# This will download the model "all-MiniLM-L6-v2" on first run.
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a collection for coding prompts
collection_name = "coding_prompt_examples"

# Ensure a clean slate by deleting and recreating the collection
try:
    client.delete_collection(name=collection_name)
except:
    pass # Ignore if collection doesn't exist

collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=ef
)

# Example successful coding prompts (to be used as few-shot examples)
# We will create two sets of examples for two different users to test filtering.

# --- User 1: 'default_user' Examples ---
prompts_user1 = [
    {
        "id": "p1_u1",
        "prompt": "As a Python developer, write a function `fibonacci(n)` that returns the nth Fibonacci number using dynamic programming. The function must include a docstring explaining the time and space complexity. The output should only be the Python code block.",
        "metadata": {"domain": "Python", "task": "Algorithm", "complexity": "Medium", "user_id": "default_user"}
    },
    {
        "id": "p2_u1",
        "prompt": "As a React expert, create a functional component `UserProfile` that accepts `user` object as props and displays the user's name and email. Use TypeScript interfaces for the props. The output should be a single JSX/TSX file content.",
        "metadata": {"domain": "JavaScript/React", "task": "Frontend", "complexity": "Easy", "user_id": "default_user"}
    }
]

# --- User 2: 'advanced_user' Examples ---
prompts_user2 = [
    {
        "id": "p3_u2",
        "prompt": "As a DevOps engineer, write a Kubernetes Deployment YAML for a microservice named 'order-processor'. The deployment must use a rolling update strategy and have 3 replicas. The output must be the raw YAML content.",
        "metadata": {"domain": "DevOps", "task": "Infrastructure", "complexity": "Hard", "user_id": "advanced_user"}
    },
    {
        "id": "p4_u2",
        "prompt": "As a Rust programmer, implement a thread-safe, lock-free MPSC queue using standard library primitives. The output should be the complete Rust code block.",
        "metadata": {"domain": "Rust", "task": "Concurrency", "complexity": "Expert", "user_id": "advanced_user"}
    }
]

all_prompts = prompts_user1 + prompts_user2

# Add the prompts to the collection
collection.add(
    documents=[p["prompt"] for p in all_prompts],
    metadatas=[p["metadata"] for p in all_prompts],
    ids=[p["id"] for p in all_prompts]
)

print(f"Successfully initialized ChromaDB collection '{collection_name}' with {len(all_prompts)} examples.")
print("The database is now ready for use in the prompt optimization agent with multi-user support.")
