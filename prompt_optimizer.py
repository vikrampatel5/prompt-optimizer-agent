import os
import sys
import argparse
from typing import Annotated, TypedDict, List

# LangChain/LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model


# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

# --- Configuration ---
LLM_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "coding_prompt_examples"

load_dotenv(os.path.join(".", ".env"), override=True)

# --- State Definition for LangGraph ---
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    - initial_prompt: The user's original, unoptimized prompt.
    - current_prompt: The prompt being refined in the current iteration.
    - refinement_history: A list of previous prompts and the LLM's feedback/reasoning.
    - iteration: The current refinement iteration count.
    - max_iterations: The maximum number of refinement steps allowed.
    - retrieved_examples: List of relevant examples from the vector DB.
    - user_id: The ID of the current user for context filtering.
    """
    initial_prompt: str
    current_prompt: str
    refinement_history: Annotated[List[str], lambda x, y: x + y]
    iteration: int
    max_iterations: int
    retrieved_examples: List[str]
    user_id: str

# --- Components ---

# 1. LLM Setup
llm = init_chat_model(model="google_genai:gemini-2.5-flash", temperature=0.0)

# 2. Vector DB Setup
def get_retriever(user_id: str):
    """Initializes ChromaDB client and returns the collection, filtered by user_id."""
    try:
        client = chromadb.CloudClient(
            api_key=os.environ['CHROMA_API_KEY'],
            tenant=os.environ['CHROMA_TENANT'],
            database='dev'
        )
        # Use the same embedding function as in setup_db.py
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)

# --- Graph Nodes (Functions) ---

def retrieve_examples(state: AgentState) -> AgentState:
    """Retrieves relevant coding prompt examples from the vector DB, filtered by user_id."""
    """Retrieves relevant coding prompt examples from the vector DB."""
    user_id = state.get("user_id", "default_user")
    print("--- Retrieving Examples from Vector DB ---")
    collection = get_retriever(user_id)
    
    # Use the initial prompt for the most relevant examples
    query = state["initial_prompt"]
    
    results = collection.query(
        query_texts=[query],
        n_results=2,
        where={"user_id": user_id}
    )
    
    # Format the retrieved documents for the LLM
    retrieved_examples = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        example = f"DOMAIN: {metadata.get('domain', 'N/A')}\nTASK: {metadata.get('task', 'N/A')}\nPROMPT: {doc}"
        retrieved_examples.append(example)
        
    print(f"Found {len(retrieved_examples)} relevant examples.")
    
    return {**state, "retrieved_examples": retrieved_examples}

def refine_prompt(state: AgentState) -> AgentState:
    """Uses the LLM to iteratively refine the current prompt."""
    print(f"\n--- Refining Prompt (Iteration {state['iteration']}/{state['max_iterations']}) ---")
    
    # Context from retrieved examples
    examples_context = "\n---\n".join(state["retrieved_examples"])
    
    # History for context
    history_context = "\n".join(state["refinement_history"])
    
    # Prompt for the Refinement LLM
    refinement_prompt_template = """
    You are an expert Prompt Engineer specializing in **coding-specific** prompts.
    Your goal is to take a user's initial prompt and iteratively refine it into a highly structured, clear, and effective prompt for a Code Generation LLM.
    
    **Optimization Criteria:**
    1.  **Clarity:** The task must be unambiguous.
    2.  **Role:** Clearly define the LLM's persona (e.g., "As a Python developer...").
    3.  **Constraints:** Specify language, libraries, output format (e.g., "Output only the code block," "Use TypeScript").
    4.  **Completeness:** Include all necessary context (e.g., table schemas, function signatures).
    5.  **Structure:** Use clear sections or formatting (e.g., bolding, bullet points).
    
    **Instructions:**
    1.  Review the **INITIAL PROMPT** and the **SUCCESSFUL EXAMPLES**.
    2.  If this is not the first iteration, review the **REFINEMENT HISTORY** to avoid repeating mistakes.
    3.  Generate a **NEW OPTIMIZED PROMPT** that incorporates the best practices from the examples and addresses any weaknesses in the current prompt.
    4.  Provide a brief **REASONING** for your changes.
    
    **SUCCESSFUL EXAMPLES (for structure and best practices):**
    {examples_context}
    
    **REFINEMENT HISTORY:**
    {history_context}
    
    **CURRENT PROMPT (to be refined):**
    {current_prompt}
    
    **OUTPUT FORMAT (STRICTLY ADHERE TO THIS):**
    You MUST output only two lines, starting with REASONING: and NEW_PROMPT:
    
    REASONING: [Your brief explanation of the changes]
    NEW_PROMPT: [The full, optimized prompt]
    """
    
    prompt_template = ChatPromptTemplate.from_template(refinement_prompt_template)
    
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Prepare the input for the chain
    input_data = {
        "examples_context": examples_context,
        "history_context": history_context,
        "current_prompt": state["current_prompt"]
    }
    
    # Invoke the chain
    response = chain.invoke(input_data)
    
    print(f"\n--- Raw LLM Response ---\n{response}\n------------------------")
    
    # Parse the response
    reasoning = "No reasoning provided."
    new_prompt = state["current_prompt"]
    
    try:
        # Find the REASONING and NEW_PROMPT markers
        reasoning_match = next((line for line in response.split('\n') if line.strip().startswith("REASONING:")), None)
        new_prompt_match = next((line for line in response.split('\n') if line.strip().startswith("NEW_PROMPT:")), None)
        
        if reasoning_match:
            reasoning = reasoning_match.replace("REASONING:", "").strip()
            
        if new_prompt_match:
            new_prompt = new_prompt_match.replace("NEW_PROMPT:", "").strip()
            
    except Exception as e:
        reasoning = f"Parsing failed: {e}. Using current prompt."
        new_prompt = state["current_prompt"]

    print(f"Reasoning: {reasoning}")
    
    # Update state
    new_history = state["refinement_history"] + [f"Iteration {state['iteration']} Reasoning: {reasoning}", f"Iteration {state['iteration']} Prompt: {new_prompt}"]
    
    return {
        **state,
        "current_prompt": new_prompt,
        "refinement_history": new_history,
        "iteration": state["iteration"] + 1
    }

# --- Graph Conditional Edge ---
def should_continue(state: AgentState) -> str:
    """Determines whether to continue refining or stop."""
    if state["iteration"] > state["max_iterations"]:
        print("\n--- Max iterations reached. Stopping refinement. ---")
        return "end"

    return "continue"

# --- Main Graph Construction ---

def build_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve_examples)
    workflow.add_node("refine", refine_prompt)
    
    # Set the entry point
    workflow.set_entry_point("retrieve")
    
    # Define the edges
    # 1. After retrieval, move to refinement
    workflow.add_edge("retrieve", "refine")
    
    # 2. After refinement, check if we should continue
    workflow.add_conditional_edges(
        "refine",
        should_continue,
        {
            "continue": "refine", # Loop back to refine
            "end": END            # End the graph
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    return app

# --- CLI Interface ---

def main():
    """Main function to run the CLI tool."""
    parser = argparse.ArgumentParser(description="AI Agent for Iterative Prompt Optimization.")
    parser.add_argument("--prompt", type=str, help="The initial prompt to be optimized.")
    parser.add_argument("--iterations", type=int, default=1, help="Maximum number of refinement iterations (default: 3).")
    parser.add_argument("--user_id", type=str, default="default_user", help="The ID of the user for context filtering (default: default_user).")

    args = parser.parse_args()
    
    # Initial State
    initial_state = AgentState(
        initial_prompt=args.prompt,
        current_prompt=args.prompt,
        refinement_history=[],
        iteration=1,
        max_iterations=args.iterations,
        retrieved_examples=[],
        user_id=args.user_id
    )
    
    print("==================================================")
    print("  AI Prompt Optimizer Agent (Coding Specific)  ")
    print("==================================================")
    print(f"Initial Prompt: {args.prompt}")
    print(f"Max Iterations: {args.iterations}")
    print(f"User: {args.user_id}")
    print("--------------------------------------------------")
    
    # Build and run the graph
    app = build_graph()
    
    # The graph will run until the 'should_continue' node returns 'end'
    final_state = None
    for state in app.stream(initial_state):
        # The state is a dictionary where the key is the node name
        # and the value is the state update from that node.
        final_state = list(state.values())[0]
        
    # Final Output
    print("\n==================================================")
    print("             Optimization Complete             ")
    print("==================================================")
    print("\n--- FINAL OPTIMIZED PROMPT ---")
    print(final_state["current_prompt"])
    print("\n--- REFINEMENT HISTORY ---")
    print("\n".join(final_state["refinement_history"]))
    print("==================================================")

if __name__ == "__main__":
    main()