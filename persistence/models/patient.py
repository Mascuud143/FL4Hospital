from __future__ import annotations

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class Patient(Base):
    """
    Represents a person.
    Stable identity only — all stay-specific data lives in Admission.
    """

    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, autoincrement=True)

    # ---- Identity / mostly stable attributes ----
    name = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    ethnicity = Column(String, nullable=True)

    # ---- Relationships ----
    admissions = relationship(
        "Admission",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    assignments = relationship(
        "RoomAssignment",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    comfort_preferences = relationship(
        "ComfortPreference",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Patient(patient_id={self.patient_id}, name={self.name})>"