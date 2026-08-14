"""Schémas Pydantic du contrat API Sanad AI (JSON In -> Markdown streamé).

La requête entrante est validée par Pydantic. La réponse, elle, est un flux
Server-Sent Events (SSE) de Markdown pur — plus aucune validation stricte de
JSON côté LLM (voir main.py).
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Corps de la requête entrante."""

    query: str = Field(
        ..., min_length=3, max_length=4000, description="Question sur l'immobilier marocain"
    )
    country: Optional[str] = Field(
        None, description="Pays de résidence du MRE (personnalisation de la réponse)"
    )
