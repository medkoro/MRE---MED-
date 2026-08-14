"""Sanad AI — API FastAPI de l'agent juridique immobilier MRE + interface web.

TROIS univers servis par le même backend :

1) API moderne (streaming SSE, Markdown pur) :
   - POST /api/v1/chat/real-estate : RAG ChromaDB -> prompt -> OpenRouter
     (Poolside Laguna S 2.1 free) en streaming, tokens relayés token par token.

2) Ancienne interface (exigence : conserver templates/ et static/ tels quels) :
   - Jinja2Templates + StaticFiles monté sur /static.
   - GET /, /chat, /observatory, /login, /register, /admin : rendu des templates
     historiques via un `url_for` compatible Flask.
   - POST /api/chat et /api/observatory-chat : endpoints de compatibilité pour
     le JavaScript existant (le frontend n'est PAS modifié).
     /api/chat bufferise le flux SSE puis renvoie le JSON attendu (Markdown -> HTML).

3) Observatoire des Talents MRE (portage Flask -> FastAPI) :
   - GET /observatory : liste les profils réels en base (SQLAlchemy pur).
   - POST /api/observatory-chat : agent LangChain tool-calling (OpenRouter)
     qui cherche dans la base et retourne les meilleurs matchs.
   - POST /admin/talents/sync : déclenche un cycle de découverte à la demande.
   - Un cycle de sync tourne aussi automatiquement toutes les 24h en tâche de
     fond (thread daemon, ne bloque pas l'event loop).
   - Authentification admin par session cookie signée (voir auth.py) :
     POST /login, POST /register, GET /logout, routes /admin/* protégées.

Format SSE (endpoint /api/v1/chat/real-estate) :
    data: {"type": "sources", "sources": [...]}
    data: {"type": "token", "content": "..."}
    data: {"type": "done"}
    data: {"type": "error", "error": "..."}

Lancement :  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from config import get_settings
from models import ChatRequest
from rag_engine import get_rag_engine

# ---------------------------------------------------------------------------
# Observatoire des Talents MRE
# ---------------------------------------------------------------------------
from database import Post, User, get_db, init_db
from auth import get_current_user, hash_password, verify_password, flash
from talent_sync import sync_talents_dynamic
from observatoire.observatory_agent import build_observatory_agent
from observatoire.scheduler import start_daily_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("sanad.ai")

settings = get_settings()

# ---------------------------------------------------------------------------
# Interface web : templates/ + static/ (préservés, non modifiés)
# ---------------------------------------------------------------------------
templates = Jinja2Templates(directory="templates")

# Cartographie des endpoints Flask historiques vers les routes FastAPI actuelles
_ENDPOINT_URLS = {
    "home": "/",
    "chat_page": "/chat",
    "observatory": "/observatory",
    "login": "/login",
    "register": "/register",
    "logout": "/logout",
    "admin_dashboard": "/admin",
    "add_post": "/admin/add",
    "toggle_post": "/admin/toggle",
    "delete_post": "/admin/delete",
    "sync_talents_admin": "/admin/talents/sync",
}


def url_for(endpoint: str, **values: str) -> str:
    """Équivalent de `flask.url_for` pour les templates Jinja historiques."""
    if endpoint == "static":
        return f"/static/{values.get('filename', '')}"
    base = _ENDPOINT_URLS.get(endpoint, "#")
    if values:
        query = "&".join(f"{key}={val}" for key, val in values.items())
        return f"{base}?{query}"
    return base


templates.env.globals["url_for"] = url_for


@pass_context
def _get_flashed_messages(context, with_categories=False):
    """Équivalent de flask.get_flashed_messages() : lit et vide les messages
    stockés en session par auth.flash()."""
    request = context["request"]
    flashes = request.session.pop("_flashes", [])
    if with_categories:
        return [tuple(f) for f in flashes]
    return [f[1] for f in flashes]


templates.env.globals["get_flashed_messages"] = _get_flashed_messages

# ----- Secteurs RAG immobilier (page d'accueil / chat) -----
SECTOR_LABELS = {
    "real_estate": "Immobilier",
    "agriculture": "Agriculture",
    "industry": "Industrie",
    "tourism": "Tourisme",
}
SECTORS = list(SECTOR_LABELS.keys())

# Code secteur frontend -> dossier data/<secteur>/ (métadonnée ChromaDB).
_SECTOR_CODE_TO_FOLDER = {
    "real_estate": "immobilier",
    "agriculture": "agriculture",
    "industry": "industrie",
    "tourism": "tourisme",
    "investissement": "investissement",
    "finance": "finance",
    "douane": "douane",
}

# ----- Secteurs Observatoire des Talents MRE (indépendants des secteurs RAG) -----
OBSERVATORY_DOMAIN_LABELS = {
    "tech": "Technologie",
    "health": "Santé",
    "education": "Éducation",
    "agriculture": "Agriculture",
    "industry": "Industrie",
    "finance": "Finance",
    "creative": "Créatif",
    "social": "Social",
    "other": "Autre",
}


class PostView:
    """Objet compatible avec l'ancien modèle SQLAlchemy `Post` pour les templates
    (utilisé uniquement là où aucune donnée réelle n'existe encore, ex: page d'accueil)."""

    def __init__(
        self,
        title: str,
        country: str | None = None,
        sector: str = "real_estate",
        years_experience: int | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> None:
        self.id = 0
        self.title = title
        self.country = country
        self.sector = sector
        self.years_experience = years_experience
        self.tags = tags or []
        self.description = description
        self.is_active = True
        self.created_at = datetime.utcnow()

    def initials(self) -> str:
        return "".join(word[0] for word in self.title.split()[:2]).upper()

    def sector_label(self) -> str:
        return SECTOR_LABELS.get(self.sector, self.sector)

    def tags_list(self) -> list[str]:
        return self.tags


# ---------------------------------------------------------------------------
# Prompt système (Markdown pur, citations obligatoires)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """Tu es **Sanad AI**, agent juridique et fiscal spécialisé en immobilier marocain pour les Marocains Résidant à l'Étranger (MRE).

Tu réponds UNIQUEMENT à partir du contexte juridique fourni ci-dessous (extraits de lois marocaines : CGI, Loi 18-00, Loi 44-00 VEFA, Dahir de la Conservation Foncière, etc.).

RÈGLES STRICTES :
1. Cite chaque affirmation avec sa source : [Source: nom_du_document, page X].
2. N'invente JAMAIS d'article, de taux, de montant ou de restriction absents du contexte.
3. Si le contexte ne contient pas l'information, réponds honnêtement que cette information ne figure pas dans ta base de connaissances.
4. Hors périmètre (droit du travail, fiscalité étrangère, etc.) : réoriente poliment vers l'immobilier marocain.

FORMAT DE SORTIE :
- Markdown pur et lisible : titres (##), listes à puces, chiffres clés en **gras**.
- Pas de JSON, pas de blocs de code, pas de balises XML. Uniquement du Markdown.

CONTEXTE JURIDIQUE (sources récupérées par le RAG) :
{context}
"""

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_event(payload: dict) -> str:
    """Sérialise un événement SSE au format standard `data: <json>\\n\\n`."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie : DB Observatoire, client HTTP partagé, worker de sync périodique.

    Le sync automatique (toutes les 24h) ne démarre QUE si ENABLE_DAILY_SYNC=true
    dans l'environnement (à activer en production uniquement). En local/dev,
    la synchronisation se déclenche uniquement via le bouton "Lancer la sync"
    du dashboard admin (POST /admin/talents/sync), qui n'est pas concerné par
    ce réglage."""
    # ----- Observatoire : DB + scheduler -----
    init_db()
    app.state.stop_sync_event = None
    if settings.ENABLE_DAILY_SYNC:
        app.state.stop_sync_event = start_daily_sync(sync_talents_dynamic)  # toutes les 24h, thread daemon
        logger.info("Sync automatique (24h) ACTIVÉ (ENABLE_DAILY_SYNC=true).")
    else:
        logger.info("Sync automatique (24h) DÉSACTIVÉ (mode dev). Utilisez le bouton admin pour synchroniser manuellement.")

    # ----- RAG juridique : client HTTP OpenRouter partagé -----
    app.state.http = httpx.AsyncClient(
        base_url=settings.OPENROUTER_BASE_URL,
        timeout=httpx.Timeout(120.0, connect=10.0),  # read timeout large : long flux SSE
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    logger.info(
        "Sanad AI démarré (modèle : %s, collection : %s, agent Observatoire : %s)",
        settings.OPENROUTER_MODEL, settings.CHROMA_COLLECTION, settings.OPENROUTER_AGENT_MODEL,
    )
    yield
    if app.state.stop_sync_event is not None:
        app.state.stop_sync_event.set()
    await app.state.http.aclose()
    logger.info("Sanad AI arrêté.")


app = FastAPI(
    title="Sanad AI — Agent Immobilier MRE",
    description="RAG juridique (ChromaDB + bge-m3) orchestré vers Poolside Laguna S 2.1 via OpenRouter. "
                "Réponses en streaming SSE, Markdown pur. Observatoire des Talents MRE (agents "
                "LangChain tool-calling via OpenRouter). Ancienne interface web servie telle quelle.",
    version="1.3.0",
    lifespan=lifespan,
)

# Session cookie signée -- nécessaire pour request.session (auth admin + flash messages)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Fichiers statiques de l'ancienne interface (images, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Filet de sécurité : aucune trace de stack leakée au client."""
    logger.exception("Erreur non gérée sur %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du serveur."},
    )


# ---------------------------------------------------------------------------
# Pipeline LLM partagé (streaming SSE) — RAG juridique, inchangé
# ---------------------------------------------------------------------------
async def _stream_llm(
    prompt: str,
    user_content: str,
    session_id: str,
    sources: list[dict],
) -> AsyncGenerator[str, None]:
    """Générateur SSE : relaye les tokens du LLM vers le client en temps réel."""
    yield sse_event({"type": "sources", "sources": sources})

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": True,
    }

    try:
        async with app.state.http.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code == 429:
                logger.warning("Session %s : quota atteint (429).", session_id)
                yield sse_event({"type": "error", "error": "Quota du modèle gratuit atteint. Réessayez plus tard."})
                return
            if response.status_code >= 400:
                logger.error("Session %s : OpenRouter %s -> %s",
                             session_id, response.status_code, (await response.aread())[:300])
                yield sse_event({"type": "error", "error": f"Le fournisseur LLM a répondu {response.status_code}."})
                return

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    token = chunk["choices"][0]["delta"].get("content") or ""
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if token:
                    yield sse_event({"type": "token", "content": token})

    except httpx.TimeoutException as exc:
        logger.error("Session %s : timeout OpenRouter (%s)", session_id, exc)
        yield sse_event({"type": "error", "error": "Le fournisseur LLM a dépassé le délai de réponse."})
        return
    except httpx.HTTPError as exc:
        logger.error("Session %s : erreur HTTP OpenRouter (%s)", session_id, exc)
        yield sse_event({"type": "error", "error": "Impossible de joindre le fournisseur LLM."})
        return

    yield sse_event({"type": "done"})
    logger.info("Session %s : flux terminé.", session_id)


def _build_rag_prompt(query: str, sector: str | None = None) -> tuple[str, list[dict]]:
    """RAG : récupération filtrée + prompt système. Retourne (prompt, sources)."""
    rag = get_rag_engine()
    chunks = rag.retrieve(query, sector=sector)
    context = (
        rag.build_context(chunks)
        if chunks
        else "Aucun document pertinent trouvé dans la base de connaissances. "
             "Réponds honnêtement que l'information ne figure pas dans ta base."
    )
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    sources = [{"source": c.source, "page": c.page} for c in chunks]
    return prompt, sources


# ---------------------------------------------------------------------------
# API moderne — streaming SSE (RAG juridique, inchangé)
# ---------------------------------------------------------------------------
@app.post("/api/v1/chat/real-estate")
async def chat_real_estate(payload: ChatRequest) -> StreamingResponse:
    """Chat juridique immobilier MRE — réponse streaming SSE (Markdown pur)."""
    session_id = uuid.uuid4().hex[:12]
    logger.info("Session %s : %s", session_id, payload.query[:120].replace("\n", " "))

    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY non configurée dans le fichier .env",
        )

    prompt, sources = _build_rag_prompt(payload.query)

    user_content = payload.query
    if payload.country:
        user_content += f"\n\n(Contexte : je réside en {payload.country}.)"

    return StreamingResponse(
        _stream_llm(prompt=prompt, user_content=user_content, session_id=session_id, sources=sources),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# Compatibilité ancien frontend (JS inchangé)
# ---------------------------------------------------------------------------
class LegacyChatRequest(BaseModel):
    """Format historique : {message, sector} (JS de chat.html)."""

    message: str = Field(..., min_length=1, max_length=4000)
    sector: str | None = Field(default=None)


class LegacyObservatoryRequest(BaseModel):
    """Format historique : {message} (JS de Observatory.html)."""

    message: str = Field(..., min_length=1, max_length=500)


@app.post("/api/chat")
async def legacy_chat(payload: LegacyChatRequest) -> StreamingResponse:
    """Point d'entrée historique (chat.html) : renvoie le flux SSE BRUT,
    sans aucune bufferisation ni conversion JSON — le JavaScript du frontend
    a été adapté pour lire le ReadableStream token par token."""
    session_id = uuid.uuid4().hex[:12]
    logger.info("Session %s : %s", session_id, payload.message[:120].replace("\n", " "))

    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY non configurée dans le fichier .env",
        )

    prompt, sources = _build_rag_prompt(payload.query, sector=_SECTOR_CODE_TO_FOLDER.get(payload.sector))

    return StreamingResponse(
        _stream_llm(prompt=prompt, user_content=payload.message, session_id=session_id, sources=sources),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# Observatoire des Talents MRE — routes réelles (branchées sur la DB + l'agent)
# ---------------------------------------------------------------------------
def _search_observatory_profiles(
    db: Session,
    keywords: list[str] | None = None,
    sector: str | None = None,
    country: str | None = None,
    min_years_experience: int | None = None,
) -> list[dict]:
    """Recherche de profils dans la base, avec un score de pertinence (0-100)
    utilisé par le nouveau template Observatory.html (badge "X% match").
    Utilisée comme `search_profiles_fn` par l'agent LangChain de l'Observatoire."""
    keywords = keywords or []
    query = db.query(Post).filter_by(is_active=True)
    if sector:
        query = query.filter(Post.sector == sector)
    candidates = query.all()

    scored: list[tuple[int, Post]] = []
    for post in candidates:
        tags = [t.lower() for t in post.tags_list()]
        title_lower = post.title.lower()
        desc_lower = (post.description or "").lower()
        keyword_hits = sum(
            1 for kw in keywords
            if any(kw in t for t in tags) or kw in title_lower or kw in desc_lower
        )

        score = 0
        if keywords:
            score += (keyword_hits / len(keywords)) * 70
        else:
            score += 40

        if min_years_experience is not None and post.years_experience is not None:
            score += 20 if post.years_experience >= min_years_experience else -15
        elif min_years_experience is None:
            score += 10

        if country and post.country and country.lower() in post.country.lower():
            score += 10

        score = max(0, min(100, round(score)))
        if score > 0 and (not keywords or keyword_hits > 0 or sector or country):
            scored.append((score, post))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "id": p.id,
            "title": p.title,
            "sector_label": p.sector_label(OBSERVATORY_DOMAIN_LABELS),
            "country": p.country,
            "years_experience": p.years_experience,
            "tags": p.tags_list(),
            "match_score": score,
            "description": p.description or "",
            "source_name": p.source_name or "",
            "source_url": p.source_url or "",
            "image_url": p.image_url or "",
        }
        for score, p in scored[:15]
    ]


def _list_observatory_options(db: Session) -> dict:
    posts = db.query(Post).filter_by(is_active=True).all()
    return {
        "sectors": sorted({p.sector for p in posts if p.sector}),
        "countries": sorted({p.country for p in posts if p.country}),
    }


@app.post("/api/observatory-chat")
async def legacy_observatory_chat(
    payload: LegacyObservatoryRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Chat de recherche de profils : un agent LangChain tool-calling (OpenRouter)
    interroge la base de l'Observatoire et sélectionne les meilleurs matchs."""

    def _search(keywords=None, sector=None, country=None, min_years_experience=None):
        return _search_observatory_profiles(
            db, keywords=keywords, sector=sector, country=country,
            min_years_experience=min_years_experience,
        )

    def _list_options():
        return _list_observatory_options(db)

    try:
        executor, selected = build_observatory_agent(
            search_profiles_fn=_search,
            list_options_fn=_list_options,
        )
        result = executor.invoke({"input": payload.message})
        response_text = result.get("output", "") if isinstance(result, dict) else str(result)
    except Exception:
        logger.exception("Erreur durant l'exécution de l'agent Observatoire")
        response_text = (
            "Une erreur est survenue pendant la recherche de profils. Réessayez dans un instant."
        )
        selected = []

    return JSONResponse({
        "response": response_text or "Aucune réponse générée.",
        "criteria": {
            "keywords": [],
            "min_years_experience": None,
            "sector": None,
            "country": None,
        },
        "matches": selected,
    })


@app.post("/admin/talents/sync")
async def sync_talents_admin(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> JSONResponse:
    """Déclenche un cycle de découverte de talents à la demande (hors du
    cycle automatique de 24h) et retourne le nombre de profils créés."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Connexion requise.")
    try:
        count = sync_talents_dynamic(db)
    except Exception:
        logger.exception("Erreur durant le sync manuel des talents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur pendant la synchronisation des talents.",
        )
    return JSONResponse({"created": count})


# ---------------------------------------------------------------------------
# Interface web (templates historiques)
# ---------------------------------------------------------------------------
@app.get("/")
async def home(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Page d'accueil publique : affiche les 4 derniers profils actifs de
    l'Observatoire dans la section 'Observatoire des compétences'."""
    posts = (
        db.query(Post)
        .filter_by(is_active=True)
        .order_by(Post.created_at.desc())
        .limit(4)
        .all()
    )
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "posts": posts,
            "sector_labels": SECTOR_LABELS,
            "user": current_user,
        },
    )


@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html", context={})


@app.get("/observatory")
async def observatory(request: Request, db: Session = Depends(get_db)):
    """Rend le template historique avec les profils réels de la base."""
    posts = (
        db.query(Post)
        .filter_by(is_active=True)
        .order_by(Post.created_at.desc())
        .all()
    )
    countries = sorted({p.country for p in posts if p.country})
    return templates.TemplateResponse(
        request=request, name="Observatory.html",
        context={
            "posts": posts,
            "sector_labels": OBSERVATORY_DOMAIN_LABELS,
            "sectors": list(OBSERVATORY_DOMAIN_LABELS.keys()),
            "countries": countries,
        },
    )


# ----- Authentification admin -----

@app.get("/login")
async def login_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/admin", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={})


@app.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email.strip().lower()).first()
    if not user or not verify_password(password, user.password_hash):
        flash(request, "Email ou mot de passe incorrect.", "error")
        return RedirectResponse(url="/login", status_code=303)

    request.session["user_id"] = user.id
    flash(request, "Connexion reussie.", "success")
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse(request=request, name="register.html", context={})


@app.post("/register")
async def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email_clean = email.strip().lower()
    if len(password) < 6:
        flash(request, "Le mot de passe doit contenir au moins 6 caracteres.", "error")
        return RedirectResponse(url="/register", status_code=303)

    if db.query(User).filter(User.email == email_clean).first():
        flash(request, "Un compte existe deja avec cet email.", "error")
        return RedirectResponse(url="/register", status_code=303)

    user = User(
        email=email_clean,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()

    flash(request, "Compte administrateur cree. Vous pouvez vous connecter.", "success")
    return RedirectResponse(url="/login", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user_id", None)
    flash(request, "Deconnecte avec succes.", "success")
    return RedirectResponse(url="/", status_code=303)


# ----- Dashboard admin (protégé) -----

@app.get("/admin")
async def admin_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    posts = db.query(Post).order_by(Post.created_at.desc()).all()
    domains = list(OBSERVATORY_DOMAIN_LABELS.keys())
    domain_counts = {d: sum(1 for p in posts if p.sector == d) for d in domains}
    stats = SimpleNamespace(
        total_users=db.query(User).count(),
        total_posts=len(posts),
        active_posts=sum(1 for p in posts if p.is_active),
        inactive_posts=sum(1 for p in posts if not p.is_active),
        auto_discovered_posts=sum(1 for p in posts if p.auto_discovered),
        countries=len({p.country for p in posts if p.country}),
        with_experience=sum(1 for p in posts if p.years_experience is not None),
        domain_counts=domain_counts,
        max_domain_count=max(domain_counts.values()) if domain_counts else 1,
    )
    return templates.TemplateResponse(
        request=request, name="admin.html",
        context={
            "user": current_user,
            "stats": stats,
            "posts": posts,
            "recent": posts[:5],
            "domains": domains,
            "domain_labels": OBSERVATORY_DOMAIN_LABELS,
        },
    )


@app.post("/admin/add")
async def add_post(
    request: Request,
    title: str = Form(...),
    sector: str = Form(...),
    country: str = Form(""),
    expertise_tags: str = Form(""),
    years_experience: str = Form(""),
    description: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    if not title.strip() or sector not in OBSERVATORY_DOMAIN_LABELS:
        flash(request, "Nom et domaine sont obligatoires.", "error")
        return RedirectResponse(url="/admin", status_code=303)

    post = Post(
        title=title.strip(), sector=sector, country=country.strip() or None,
        expertise_tags=expertise_tags.strip(), description=description.strip(),
        years_experience=int(years_experience) if years_experience.isdigit() else None,
        auto_discovered=False,
    )
    db.add(post)
    db.commit()
    flash(request, "Profil talent publie.", "success")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/toggle")
async def toggle_post(
    request: Request,
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    post = db.get(Post, post_id)
    if post:
        post.is_active = not post.is_active
        db.commit()
        flash(request, "Statut mis a jour.", "success")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/delete")
async def delete_post(
    request: Request,
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    post = db.get(Post, post_id)
    if post:
        db.delete(post)
        db.commit()
        flash(request, "Annonce supprimee.", "success")
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/health")
async def health() -> dict:
    """Probe de santé : état du LLM, de la base vectorielle et de l'Observatoire."""
    rag = get_rag_engine()
    return {
        "status": "ok",
        "model": settings.OPENROUTER_MODEL,
        "llm_configured": bool(settings.OPENROUTER_API_KEY),
        "collection": settings.CHROMA_COLLECTION,
        "chroma_chunks": rag.collection_count,
        "observatory_agent_model": settings.OPENROUTER_AGENT_MODEL,
        "observatory_pool_configured": bool(settings.OPENROUTER_API_KEY_1 and settings.OPENROUTER_API_KEY_2),
        "tavily_configured": bool(settings.TAVILY_API_KEY),
    }


if __name__ == "__main__":
    import os

    import uvicorn

    # PORT : défini par la plateforme d'hébergement (7860 = Hugging Face,
    # 10000 = Render, etc.). Défaut local : 8000.
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))