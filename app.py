"""Point d'entrée Hugging Face Spaces (SDK Gradio + ZeroGPU).

L'application elle-même est FastAPI (main:app) — pas Gradio. Mais sur un
compte gratuit, Hugging Face n'accepte que le hardware ZeroGPU pour les
Spaces exécutables, et ZeroGPU exige au démarrage la présence d'au moins
une fonction décorée @spaces.GPU.

Pour une app Gradio pure, le package `spaces` envoie automatiquement le
rapport de démarrage lors de `demo.launch()`. Ici l'app est une FastAPI
sur laquelle la mini-app Gradio est montée : on émet donc ce rapport
manuellement via `spaces.zero.startup()` (pattern du Space FastAPI+ZeroGPU
Jbowyer/Hunyuan3D-2.1, en ligne), sinon le runtime répond
"No @spaces.GPU function detected during startup".

Usage :
    uvicorn main:app --reload --port 8000   # en local (sans gradio)
    python app.py                           # sur Hugging Face (port 7860)
"""

import os
import sys

import spaces

import gradio as gr

from main import app as fastapi_app


print(f"[app.py] module-level import, pid={os.getpid()} argv={sys.argv}", flush=True)


@spaces.GPU(duration=1)
def _zerogpu_probe() -> str:
    return "GPU prêt"


with gr.Blocks() as demo:
    btn = gr.Button("Initialiser le GPU")
    out = gr.Textbox()
    btn.click(fn=_zerogpu_probe, inputs=[], outputs=[out])


app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio-internal")


def _probe_port(port: int) -> None:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        print(f"[app.py] probe port {port}: FREE", flush=True)
        s.close()
    except OSError as exc:
        print(f"[app.py] probe port {port}: BUSY -> {exc}", flush=True)
        try:
            ns = os.popen("netstat -tlnp 2>/dev/null || netstat -tln 2>/dev/null").read()
            print(f"[app.py] netstat:\n{ns}", flush=True)
        except Exception:
            pass
        s.close()


_probe_port(int(os.environ.get("PORT", "7860")))
print(f"[app.py] PORT env={os.environ.get('PORT')!r}", flush=True)

# ZeroGPU : envoyer le rapport de démarrage (le runtime attend au moins une
# fonction @spaces.GPU ; en l'absence de launch() gradio, on le déclenche
# nous-mêmes). No-op hors ZeroGPU.
if getattr(spaces, "zero", None) is not None and hasattr(spaces.zero, "startup"):
    spaces.zero.startup()

print(f"[app.py] end of module, pid={os.getpid()}", flush=True)

if __name__ == "__main__":
    print(f"[app.py] __main__ pid={os.getpid()}", flush=True)
    _probe_port(int(os.environ.get("PORT", "7860")))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
