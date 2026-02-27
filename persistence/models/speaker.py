from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    DateTime,
    Index,
)
from sqlalchemy.orm import relationship

from ..database import Base


class Speaker(Base):
    __tablename__ = "speakers"

    speaker_id = Column(Integer, primary_key=True, autoincrement=True)

    # Optional intensity (e.g. 0.0–1.0 or 0–100 depending on your app)
    level = Column(Float, nullable=True)

    # One-to-one with Device
    device_id = Column(
        Integer,
        ForeignKey("devices.device_id", ondelete="CASCADE"),
        nullable=False,
        unique=False,  # allow multiple speaker records for the same device over time
        index=True,
    )

    # When this speaker state became active
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # Relationships
    device = relationship(
        "Device",
        back_populates="speaker",
        uselist=False,
    )

    __table_args__ = (
        Index("ix_speaker_device_time", "device_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Speaker("
            f"id={self.speaker_id}, "
            f"device_id={self.device_id}, "
            f"level={self.level}, "
            f"timestamp={self.timestamp}"
            f")>"
        )
