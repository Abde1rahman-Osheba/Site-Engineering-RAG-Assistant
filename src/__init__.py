"""1830 PSS Multimodal RAG - reusable pipeline package used by the notebook."""
from pathlib import Path

# Load a project-root `.env` file (API keys, HF_TOKEN, ...) into the process
# environment, if one exists - so `export`-ing keys by hand every session is
# optional. Never overwrites a variable already set in the real environment
# (override=False), and does nothing at all if no `.env` file is present -
# see `.env.example` for what it can contain. This runs once, on first import
# of the `src` package, so every entry point (notebook/scripts/tests) picks
# it up the same way.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed - .env just won't be auto-loaded; env vars/`export` still work
