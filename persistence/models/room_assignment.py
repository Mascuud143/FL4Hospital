from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class RoomAssignment(Base):
    __tablename__ = "room_assignments"

    assignment_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False, index=True)
    room_id = Column(Integer, ForeignKey("rooms.room_id"), nullable=False, index=True)

    start_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="assignments")
    room = relationship("Room", back_populates="assignments")

    def __repr__(self) -> str:
        return f"<RoomAssignment(assignment_id={self.assignment_id}, patient_id={self.patient_id}, room_id={self.room_id})>"
