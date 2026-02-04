from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class ComfortPreference(Base):
    __tablename__ = "comfort_preferences"

    comfort_pref_id = Column(Integer, primary_key=True, autoincrement=True)

    temperature = Column(Float, nullable=True)
    light_intensity = Column(Float, nullable=True)
    sound_level = Column(Float, nullable=True)
    ventilation = Column(Float, nullable=True)

    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.room_id"), nullable=True, index=True)

    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    source = Column(String, nullable=True)  # "Simulation" | "manual" | "AI"

    # Relationships
    patient = relationship("Patient", back_populates="comfort_preferences")
    room = relationship("Room", back_populates="comfort_preferences")

    def __repr__(self) -> str:
        return f"<ComfortPreference(id={self.comfort_pref_id}, patient_id={self.patient_id}, room_id={self.room_id}, source={self.source})>"
