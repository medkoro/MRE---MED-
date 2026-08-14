"""Point d'entrée Hugging Face Spaces (SDK Gradio).

L'application est FastAPI (pas Gradio) et CPU-only : embeddings locaux CPU,
LLM appelé via API. Le Space utilise donc le matériel gratuit `cpu-basic`
(configuré dans le README) — ZeroGPU est inutile ici et exigerait une
fonction décorée @spaces.GPU + une interface Gradio montée.

Hugging Face exécute `app.py`, donc ce wrapper suffit à servir toute
l'application web (templates/ + static/ + API SSE) telle quelle.

Usage :
    python app.py          # démarre uvicorn sur 0.0.0.0:7860
"""

import os

import uvicorn

from main import app as fastapi_app

_PORT = int(os.environ.get("PORT", "7860"))

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=_PORT)
