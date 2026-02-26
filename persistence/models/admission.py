# persistence/models/admission.py
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class Admission(Base):
    __tablename__ = "admissions"

    admission_id = Column(Integer, primary_key=True, autoincrement=True)

    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)

    # Optional but very useful: the "primary" room for the stay (first room)
    # Actual movements still tracked by RoomAssignment.
    initial_room_id = Column(Integer, ForeignKey("rooms.room_id", ondelete="SET NULL"), nullable=True, index=True)

    admitted_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    discharged_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # ---- per-admission changing attributes ----
    age = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True)
    current_diagnosis = Column(String, nullable=True)

    # relationships
    patient = relationship("Patient", back_populates="admissions")
    room_assignments = relationship("RoomAssignment", back_populates="admission", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_admission_patient_time", "patient_id", "admitted_at"),
    )

    def __repr__(self) -> str:
        return f"<Admission(admission_id={self.admission_id}, patient_id={self.patient_id}, admitted_at={self.admitted_at}, discharged_at={self.discharged_at})>"