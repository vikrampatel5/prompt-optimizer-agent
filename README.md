# agentic-mcp

A small prototype agent for iterative prompt optimization targeted at coding prompts. The agent uses a combination of LangChain / LangGraph, a vector DB (ChromaDB), and a chat LLM (e.g. Google Generative AI models) to retrieve successful examples and iteratively refine a user's prompt into a clearer, more structured prompt for code-generation models.

Key components
- `prompt_optimizer.py` — CLI tool that builds a LangGraph state machine to 1) retrieve relevant example prompts from a ChromaDB collection and 2) iteratively refine the user's prompt using an LLM.
- `setup_db.py` — example script to populate a ChromaDB collection (`coding_prompt_examples`) with sample prompts and metadata to use as few-shot examples.

Requirements
- Python 3.11 or newer
- See `pyproject.toml` for the main runtime dependencies (LangChain, ChromaDB, Google/OpenAI integrations, sentence-transformers, etc.).

Quickstart (Windows cmd.exe)
1. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install the project in editable mode (uses the deps in `pyproject.toml`)

```cmd
pip install -e .
```

3. Provide API keys and configuration

Create a `.env` file in the project root or set environment variables in your shell. The project expects keys for the LLM and vector DB. Suggested variables:

- GOOGLE_API_KEY (or credentials required by your chosen chat model integration)
- CHROMA_API_KEY (or equivalent) — your ChromaDB Cloud API key
- CHROMA_TENANT — your ChromaDB tenant id

Example `.env` (DO NOT commit this file):

```text
# .env (example)
GOOGLE_API_KEY=ya29.A0AR...your-google-key...
CHROMA_API_KEY=ck-...your-chroma-key...
CHROMA_TENANT=your-tenant-id
```

Important: `setup_db.py` and `prompt_optimizer.py` currently contain example/hardcoded values in the repository. Before running anything against a real ChromaDB account, review these files and replace hardcoded keys with environment-based configuration.

Initialize example vector DB data

The repository includes `setup_db.py` which populates a ChromaDB collection with a few example prompts used by the optimizer for retrieval. Run it after you set your ChromaDB credentials (or edit the script to use env vars):

```cmd
python setup_db.py
```

Run the prompt optimizer CLI

```cmd
python prompt_optimizer.py --prompt "I need to speed up my database queries in Python" --iterations 2 --user_id default_user
```

- `--prompt` — the initial unoptimized prompt
- `--iterations` — max number of refinement iterations
- `--user_id` — filters retrieved examples by user metadata (see `setup_db.py` examples)

Run the demo in `main.py`

```cmd
python main.py
```

Notes and configuration
- The prompt optimizer is opinionated for coding prompts and enforces a strict output format when asking the LLM to return reasoning and a new prompt.
- The project uses `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings. The first run will download the model weights.
- Be careful with secrets: do not commit `.env` or API keys to source control.
- If you are using ChromaDB Cloud, confirm the client initialization parameters in `prompt_optimizer.py` and `setup_db.py` match your account and prefer reading values from environment variables for safety.

Development
- Use the editable install to iterate quickly: `pip install -e .`
- Add tests and type checks as you extend the project.

Troubleshooting
- If you see errors contacting ChromaDB, verify your API key, tenant, and database settings.
- If the LLM calls fail, confirm your model configuration and API credentials (Google/OpenAI) are valid and that any client libraries are up-to-date.