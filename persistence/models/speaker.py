from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from ..database import Base


class Speaker(Base):
    __tablename__ = "speaker"

    speaker_id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(Float, nullable=True)

    device_id = Column(Integer, ForeignKey("devices.device_id"), nullable=False, unique=True, index=True)

    # Relationships
    device = relationship("Device", back_populates="speaker")

    def __repr__(self) -> str:
        return f"<Speaker(speaker_id={self.speaker_id}, device_id={self.device_id}, level={self.level})>"
