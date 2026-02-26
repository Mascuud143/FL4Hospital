from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class Medication(Base):
    __tablename__ = "medications"

    medication_id = Column(Integer, primary_key=True, autoincrement=True)

    patient_id = Column(
        Integer,
        ForeignKey("patients.patient_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    medication_time = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    route = Column(String, nullable=True)      # oral | injection | IV
    drug_name = Column(String, nullable=False)
    dose = Column(String, nullable=True)       # keep string for "500mg", "2 tablets", etc.
    status = Column(String, nullable=True)     # taken | missed

    # Relationships
    patient = relationship("Patient", back_populates="medications")

    __table_args__ = (
        Index("ix_medication_patient_time", "patient_id", "medication_time"),
        Index("ix_medication_drug", "drug_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Medication(medication_id={self.medication_id}, patient_id={self.patient_id}, "
            f"drug_name={self.drug_name}, dose={self.dose}, route={self.route}, "
            f"status={self.status}, medication_time={self.medication_time})>"
        )
    