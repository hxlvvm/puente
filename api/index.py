"""Vercel entry point: the FastAPI app, served under /api/* (see vercel.json)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.app import app  # noqa: E402,F401
