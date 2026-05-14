from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base class for all ORM models
Base = declarative_base()

# Module-level singletons (created lazily)
_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def sqlite_path_from_url(db_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return None
    raw_path = db_url[len(prefix):]
    if not raw_path:
        return None
    return Path(raw_path)


def sqlite_url_from_path(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def reset_sqlite_db(db_url: str, *, fallback_label: str) -> tuple[str, bool]:
    path = sqlite_path_from_url(db_url)
    if path is None:
        return db_url, False

    try:
        if path.exists():
            path.unlink()
        return db_url, False
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}.{fallback_label}.{stamp}{path.suffix}")
        if fallback_path.exists():
            fallback_path.unlink()
        return sqlite_url_from_path(fallback_path), True


def get_engine(db_url: str = "sqlite:///fl4hospital.db", echo: bool = False) -> Engine:
    """
    Create (or return) a global SQLAlchemy Engine for SQLite.

    Notes for SQLite:
    - check_same_thread=False helps when used from Flask / multiple threads.
    - future=True for SQLAlchemy 1.4+/2.0 style.
    """
    global _ENGINE, _SessionLocal
    if _ENGINE is None:
        _ENGINE = create_engine(
            db_url,
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False},
        )
        _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)
    return _ENGINE


def get_session() -> Session:
    """
    Return a new Session from the global sessionmaker.
    Make sure get_engine() has been called at least once.
    """
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def init_db(db_url: str = "sqlite:///fl4hospital.db", echo: bool = False) -> None:
    """
    Create all tables.
    Important: ensure models are imported so Base knows them.
    """
    engine = get_engine(db_url=db_url, echo=echo)

    # Import models so they register with Base.metadata
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Safe session scope:
      with session_scope() as s:
          ...
    Commits on success, rollbacks on exception.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
