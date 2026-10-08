from pathlib import Path
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# --------------------------------------------------
# Database URL
# --------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


# --------------------------------------------------
# Local fallback: SQLite
# --------------------------------------------------

if not DATABASE_URL:

    DATA_DIR = (
        PROJECT_ROOT
        / "data"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DATABASE_PATH = (
        DATA_DIR
        / "app.db"
    )

    DATABASE_URL = (
        f"sqlite:///"
        f"{DATABASE_PATH.as_posix()}"
    )


# --------------------------------------------------
# Fix old postgres URL format if needed
# --------------------------------------------------

if DATABASE_URL.startswith(
    "postgres://"
):

    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )


# --------------------------------------------------
# Engine
# --------------------------------------------------

if DATABASE_URL.startswith(
    "sqlite"
):

    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False
        }
    )

else:

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


# --------------------------------------------------
# Session
# --------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# --------------------------------------------------
# Base
# --------------------------------------------------

Base = declarative_base()


# --------------------------------------------------
# DB dependency
# --------------------------------------------------

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()