from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    DateTime,
    Index,
)
from sqlalchemy.orm import relationship

from ..database import Base


class Ventilation(Base):
    __tablename__ = "ventilations"

    ventilation_id = Column(Integer, primary_key=True, autoincrement=True)

    # System-controlled mode (patient does NOT set this)
    mode = Column(
        String,
        nullable=False,
        doc="heat | cool | airflow",
    )

    # Optional intensity / power level (e.g. 0.0–1.0 or device-specific scale)
    level = Column(Float, nullable=True)

    # One-to-one with Device
    device_id = Column(
        Integer,
        ForeignKey("devices.device_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # When this ventilation state became active
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # Relationships
    device = relationship(
        "Device",
        back_populates="ventilation",
        uselist=False,
    )

    __table_args__ = (
        Index("ix_ventilation_device_time", "device_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Ventilation("
            f"id={self.ventilation_id}, "
            f"device_id={self.device_id}, "
            f"mode={self.mode}, "
            f"level={self.level}, "
            f"timestamp={self.timestamp}"
            f")>"
        )
