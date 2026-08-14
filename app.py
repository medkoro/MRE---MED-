"""Point d'entrée Hugging Face Spaces (SDK Gradio + ZeroGPU).

L'application elle-même est FastAPI (main:app) — pas Gradio. Mais sur un
compte gratuit, Hugging Face n'accepte que le hardware ZeroGPU pour les
Spaces exécutables, et ZeroGPU exige au démarrage une interface Gradio avec
au moins une fonction décorée @spaces.GPU liée à un handler enregistré.

Ce fichier crée donc une mini-app Gradio "fantôme" (@spaces.GPU + bouton)
au niveau module (le runtime HF scanne les handlers de la variable `demo`),
puis la monte sur la FastAPI via gr.mount_gradio_app. Le runtime ZeroGPU
sert lui-même l'app montée sur le port 7860 : NE PAS relancer uvicorn.

Usage local :  uvicorn main:app --reload --port 8000
"""

import spaces

import gradio as gr

from main import app as fastapi_app


@spaces.GPU(duration=60)
def _gpu_probe() -> str:
    return "GPU prêt"


with gr.Blocks() as demo:
    btn = gr.Button("Initialiser le GPU")
    out = gr.Textbox()
    btn.click(fn=_gpu_probe, inputs=[], outputs=[out])


app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio-internal")
