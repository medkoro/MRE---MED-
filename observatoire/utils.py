"""Utilitaires partagés pour l'observatoire des talents."""

from __future__ import annotations


def tokenize_keywords(value: str | None) -> list[str]:
    """Convertit une chaîne de mots-clés en liste de tokens propres."""
    if not value:
        return []
    return [token.lower() for token in value.replace(",", " ").split() if token.strip()]
