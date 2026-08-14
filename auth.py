"""
Authentification pour l'Observatoire / dashboard admin.

Session cookie signee (via starlette SessionMiddleware, a ajouter dans
main.py) -- pas de JWT, pas de table de sessions cote serveur : le cookie
contient l'id utilisateur, signe avec SECRET_KEY. Suffisant pour un seul
dashboard admin peu sollicite.

Remplace flask_login (login_user/current_user/login_required) de l'ancien
app.py Flask.
"""
import bcrypt
from fastapi import Depends, Request
from sqlalchemy.orm import Session

from database import User, get_db


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """Dependance FastAPI : retourne le User connecte ou None (ne bloque
    jamais -- a la route de decider quoi faire si None)."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def flash(request: Request, message: str, category: str = "success") -> None:
    """Equivalent de flask.flash() : stocke un message dans la session,
    consomme au prochain rendu de template via get_flashed_messages()."""
    flashes = request.session.get("_flashes", [])
    flashes.append([category, message])
    request.session["_flashes"] = flashes