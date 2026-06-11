"""Entry point: .venv/bin/python -m webapp  (PORT env overrides the default)"""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=port, log_level="info")
