from __future__ import annotations

from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class Repository(Generic[T]):
    """
    Generic CRUD repository for SQLAlchemy models.
    Use specialized repositories for complex queries.
    """

    def __init__(self, model: Type[T]) -> None:
        self.model = model

    def add(self, session: Session, obj: T) -> T:
        session.add(obj)
        return obj

    def get(self, session: Session, obj_id: Any) -> Optional[T]:
        return session.get(self.model, obj_id)

    def list(self, session: Session, limit: int = 100, offset: int = 0) -> List[T]:
        return list(session.query(self.model).offset(offset).limit(limit).all())

    def delete(self, session: Session, obj: T) -> None:
        session.delete(obj)

    def update(self, session: Session, obj: T, **fields: Any) -> T:
        for k, v in fields.items():
            setattr(obj, k, v)
        session.add(obj)
        return obj
