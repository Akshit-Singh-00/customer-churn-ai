from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

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


# --------------------------------------------------
# SQLite database URL
# --------------------------------------------------

DATABASE_URL = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


# --------------------------------------------------
# Database engine
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
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
# Base model
# --------------------------------------------------

Base = declarative_base()


# --------------------------------------------------
# Database dependency
# --------------------------------------------------

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()