"""Point d'entrée Hugging Face Spaces (SDK Gradio).

Depuis juillet 2026, Hugging Face facture les Spaces Docker SDK (plan PRO).
L'option gratuite restante pour les comptes personnels est le SDK Gradio
sur le matériel ZeroGPU gratuit (jusqu'à 2 Spaces).

Ce fichier démarre le serveur FastAPI existant (main:app) sur le port 7860.
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
