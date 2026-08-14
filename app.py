"""Point d'entrée Hugging Face Spaces (SDK Gradio).

L'application elle-même est FastAPI (main:app) — pas Gradio. Mais sur un
compte gratuit, Hugging Face n'accepte que le hardware ZeroGPU pour les
Spaces exécutables, et ZeroGPU exige au démarrage une interface Gradio avec
au moins une fonction décorée @spaces.GPU liée à un handler.

Ce fichier monte donc une mini-app Gradio "fantôme" (@spaces.GPU + bouton)
sur la FastAPI existante via gr.mount_gradio_app, puis démarre uvicorn.
La FastAPI sert toute l'application web (templates/ + static/ + API SSE)
et la mini-app Gradio ne sert qu'à satisfaire la détection ZeroGPU.

Usage :
    python app.py          # démarre uvicorn sur 0.0.0.0:7860
"""

import os

import uvicorn

from main import app as fastapi_app


def build_app():
    """Assemble la FastAPI + mini-app Gradio exigée par ZeroGPU.

    Retourne l'app FastAPI seule si gradio/spaces ne sont pas installés
    (ex. exécution locale), la FastAPI montée avec Gradio sinon.
    """
    try:
        import gradio as gr
        import spaces

        @spaces.GPU(duration=60)
        def _gpu_probe() -> str:
            return "GPU prêt"

        with gr.Blocks() as demo:
            btn = gr.Button("Initialiser le GPU")
            out = gr.Textbox()
            btn.click(fn=_gpu_probe, inputs=[], outputs=[out])

        return gr.mount_gradio_app(fastapi_app, demo, path="/gradio-internal")
    except Exception as exc:  # noqa: BLE001 — fallback local
        print(f"[app.py] Gradio/ZeroGPU indisponible ({exc}), app FastAPI seule.")
        return fastapi_app


app = build_app()

_PORT = int(os.environ.get("PORT", "7860"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_PORT)
