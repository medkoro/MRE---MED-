"""Configuration centralisée du backend Sanad AI (pydantic-settings).

Les variables sont lues depuis le fichier `.env` à la racine du projet.
Chaque valeur peut être surchargée par une variable d'environnement du même nom.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration du microservice Sanad AI (agent immobilier MRE)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- LLM (OpenRouter) -----
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="Clé API OpenRouter (https://openrouter.ai/keys). Vide = chat désactivé avec 503.",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Endpoint OpenAI-compatible d'OpenRouter.",
    )
    OPENROUTER_MODEL: str = Field(
        default="poolside/laguna-s-2.1:free",
        description="Modèle LLM via OpenRouter : Poolside Laguna S 2.1 (gratuit, 262K contexte).",
    )
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0)
    LLM_MAX_TOKENS: int = Field(default=4096, gt=0, le=32768)
    SECRET_KEY: str = Field(default="dev_secret_key_change_in_production")
    # ----- Observatoire des Talents MRE (agents LangChain) -----
    OPENROUTER_API_KEY_1: str = Field(
        default="",
        description="1ère clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_API_KEY_2: str = Field(
        default="",
        description="2ème clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_EXTRACTOR_MODEL: str = Field(
        default="poolside/laguna-s-2.1:free",
        description="Modèle OpenRouter gratuit pour extraire/valider un profil de talent (JSON structuré).",
    )
    OPENROUTER_AGENT_MODEL: str = Field(
        default="poolside/laguna-s-2.1:free",
        description="Modèle OpenRouter gratuit pour les agents tool-calling (scout + observatory). "
        "Doit supporter le tool-calling sur OpenRouter.",
    )
    TAVILY_API_KEY: str = Field(
        default="",
        description="Clé API Tavily pour la découverte web de talents MRE.",
    )

    # ----- RAG / ChromaDB -----
    CHROMA_PERSIST_DIR: str = Field(default="chroma_db", description="Dossier de persistance ChromaDB.")
    CHROMA_COLLECTION: str = Field(
        default="immobilier_mre",
        description="Collection dédiée au droit immobilier marocain pour les MRE.",
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Embeddings locaux multilingues FR/AR (384 dimensions, ~4x plus rapide que bge-m3 sur CPU). "
        "Pour plus de précision, remettre BAAI/bge-m3 (1024 dims) et purger chroma_db/.",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Taille de lot d'embedding (CPU : 64-128 accélère nettement).",
    )
    EMBEDDING_DEVICE: str = Field(
        default="auto",
        description="Périphérique d'embedding : 'auto' (cuda si dispo, sinon cpu), 'cpu', 'cuda'.",
    )
    TOP_K: int = Field(default=6, ge=1, le=20, description="Nombre de chunks récupérés par le RAG.")
    SIMILARITY_THRESHOLD: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Seuil de distance cosine ChromaDB (plus bas = plus proche). "
        "Les chunks au-delà du seuil sont écartés : anti-hallucination. "
        "Observé sur bge-m3 : pertinence forte < 0.45, moyenne < 0.60.",
    )

    # ----- API -----
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:8000"],
        description="Origines CORS autorisées (format JSON dans .env, ex: [\"http://x\"]).",
    )
    ENABLE_DAILY_SYNC: bool = Field(
        default=False,
        description="Activer la synchronisation automatique des talents toutes les 24h."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuration (recalculé à chaud pour les tests)."""
    return Settings()


def resolve_device(device: str | None = None) -> str:
    """Résout 'auto'/'vide' vers 'cuda' si un GPU est disponible (sinon 'cpu').

    Si explicitement demandé ('cuda'), vérifie que CUDA est réellement disponible
    avant de l'utiliser (sinon retombe sur 'cpu')."""
    requested = (device or "").strip().lower()
    if requested in ("", "auto", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"