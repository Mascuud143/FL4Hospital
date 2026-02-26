from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class Visit(Base):
    __tablename__ = "visits"

    visit_id = Column(Integer, primary_key=True, autoincrement=True)

    patient_id = Column(
        Integer,
        ForeignKey("patients.patient_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    visit_time = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    body_temperature = Column(Float, nullable=True)   # °C
    blood_pressure = Column(String, nullable=True)    # e.g. "120/80"
    symptoms = Column(String, nullable=True)          # free text / tags

    # Relationships
    patient = relationship("Patient", back_populates="visits")

    __table_args__ = (
        Index("ix_visit_patient_time", "patient_id", "visit_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<Visit(visit_id={self.visit_id}, patient_id={self.patient_id}, "
            f"visit_time={self.visit_time}, body_temperature={self.body_temperature}, "
            f"blood_pressure={self.blood_pressure})>"
        )