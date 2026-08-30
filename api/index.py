"""Vercel Python entrypoint.

Vercel's Python runtime scans a fixed list of paths (api/index.py among
them) for an ASGI/WSGI `app` object. The actual application lives in
llm_gateway/api.py; this just re-exports it from a path Vercel finds.
"""

from llm_gateway.api import app

__all__ = ["app"]
