from __future__ import annotations

from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.orm import relationship

from ..database import Base


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, autoincrement=True)

    age = Column(Integer, nullable=True)
    name = Column(String, nullable=True)

    weight = Column(Float, nullable=True)
    gender = Column(String, nullable=True)
    height = Column(Float, nullable=True)

    ethnicity = Column(String, nullable=True)
    current_diagnosis = Column(String, nullable=True)

    admission_date = Column(Date, nullable=True)
    release_date = Column(Date, nullable=True)

    # Relationships
    assignments = relationship("RoomAssignment", back_populates="patient", cascade="all, delete-orphan")
    comfort_preferences = relationship("ComfortPreference", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Patient(patient_id={self.patient_id}, name={self.name})>"
