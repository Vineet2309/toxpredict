"""
run.py — Render.com entry point
Run with: python run.py
"""
import os
import uvicorn

port = int(os.environ.get("PORT", 8000))
print(f"[run.py] Binding to port {port}", flush=True)

uvicorn.run(
    "api.main:app",
    host="0.0.0.0",
    port=port,
    log_level="info",
)
