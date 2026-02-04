from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from ..database import Base


class Ventilation(Base):
    __tablename__ = "ventilation"

    ventilation_id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(Float, nullable=True)

    device_id = Column(Integer, ForeignKey("devices.device_id"), nullable=False, unique=True, index=True)

    # Relationships
    device = relationship("Device", back_populates="ventilation")

    def __repr__(self) -> str:
        return f"<Ventilation(ventilation_id={self.ventilation_id}, device_id={self.device_id}, level={self.level})>"
