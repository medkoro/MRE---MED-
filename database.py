import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./instance/talents.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Post(Base):
    __tablename__ = "post"

    id = Column(Integer, primary_key=True)
    title = Column(String(150), nullable=False)
    sector = Column(String(50), nullable=False)
    country = Column(String(100))
    expertise_tags = Column(String(200))
    years_experience = Column(Integer)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source_url = Column(String(500))
    source_name = Column(String(150))
    image_url = Column(String(500))
    auto_discovered = Column(Boolean, default=False)

    _DEFAULT_SECTOR_LABELS = {
        "tech": "Technologie", "health": "Santé", "education": "Éducation",
        "agriculture": "Agriculture", "industry": "Industrie", "finance": "Finance",
        "creative": "Créatif", "social": "Social", "other": "Autre",
    }

    def sector_label(self, labels: dict | None = None) -> str:
        labels = labels or self._DEFAULT_SECTOR_LABELS
        return labels.get(self.sector, self.sector)

    def tags_list(self) -> list[str]:
        if not self.expertise_tags:
            return []
        return [t.strip() for t in self.expertise_tags.split(",") if t.strip()]

    def initials(self) -> str:
        parts = self.title.split()
        letters = "".join(p[0] for p in parts[:2] if p)
        return letters.upper() or "?"


def init_db():
    os.makedirs("instance", exist_ok=True)
    Base.metadata.create_all(bind=engine)  # crée Post ET User


def create_default_admin_if_missing(default_email: str, default_password: str):
    """A appeler explicitement au demarrage (pas automatique) si tu veux un
    admin de secours. Ne fait rien si un compte existe deja avec cet email."""
    import bcrypt
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == default_email).first():
            return
        hashed = bcrypt.hashpw(default_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db.add(User(email=default_email, password_hash=hashed, country="Maroc", sector_interest="real_estate"))
        db.commit()
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()