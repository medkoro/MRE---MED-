"""Moteur RAG de l'agent immobilier MRE (Sanad AI).

Responsabilités :
    - Interroger la collection ChromaDB alimentée par `ingest.py`.
    - Filtrer par métadonnées (`sector`, optionnel) AVANT l'injection
      dans le contexte : seule la donnée du secteur demandé est injectée.
    - Appliquer un seuil de similarité : les chunks faibles sont écartés
      (anti-hallucination).
    - Formater le contexte Markdown injecté dans le prompt final.

Note d'architecture : la collection est peuplée avec des embeddings explicites
(bge-m3, 1024 dimensions). Il faut donc interroger avec `query_embeddings`,
jamais avec `query_texts` (la fonction d'embedding par défaut de ChromaDB
produirait des vecteurs de dimensions incompatibles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings, get_settings, resolve_device

logger = logging.getLogger("sanad.rag")


@dataclass
class RetrievedChunk:
    """Chunk récupéré par le RAG (usage interne au moteur)."""

    source: str
    page: int | None
    content: str


class ImmobilierRAGEngine:
    """Moteur de recherche sémantique sur la base juridique immobilier MRE."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        device = resolve_device(settings.EMBEDDING_DEVICE)
        logger.info("Embeddings sur périphérique '%s' (EMBEDDING_DEVICE=%s).",
                    device, settings.EMBEDDING_DEVICE)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' prête (%d chunk(s)).",
            settings.CHROMA_COLLECTION,
            self._collection.count(),
        )

    @property
    def collection_count(self) -> int:
        return self._collection.count()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sector: str | None = None,
    ) -> list[RetrievedChunk]:
        """Recherche sémantique + filtre métadonnées optionnel + seuil de similarité.

        `sector` : si fourni (ex. "immobilier", "finance", "tourisme"), ne retient
        que les chunks de ce secteur. Sinon (défaut), recherche dans TOUS les
        secteurs présents dans la collection (ingest.py déduit le secteur du
        dossier data/<secteur>/).
        """
        k = top_k or self.settings.TOP_K
        if self._collection.count() == 0:
            logger.warning("Collection vide : lancez `python ingest.py` pour alimenter la base.")
            return []

        query_vector = self._embeddings.embed_query(query)

        query_kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if sector:
            query_kwargs["where"] = {"sector": sector}

        results = self._collection.query(**query_kwargs)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Distance cosine ChromaDB : plus bas = plus proche (1 - cos).
            if dist > self.settings.SIMILARITY_THRESHOLD:
                continue
            chunks.append(
                RetrievedChunk(
                    source=meta.get("source", "inconnu"),
                    page=meta.get("page"),
                    content=doc,
                )
            )

        logger.info("RAG : %d source(s) retenue(s) sur %d candidat(s).", len(chunks), len(documents))
        return chunks

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formate les chunks en contexte Markdown sourcé pour le prompt LLM."""
        blocks = [
            f"[Document: {chunk.source}, page {chunk.page or 'N/A'}]\n{chunk.content}"
            for chunk in chunks
        ]
        return "\n\n".join(blocks)


_engine: ImmobilierRAGEngine | None = None


def get_rag_engine() -> ImmobilierRAGEngine:
    """Singleton paresseux du moteur RAG (chargement lourd : bge-m3)."""
    global _engine
    if _engine is None:
        _engine = ImmobilierRAGEngine(get_settings())
    return _engine
