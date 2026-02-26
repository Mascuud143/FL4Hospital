from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class RoomAssignment(Base):
    __tablename__ = "room_assignments"

    assignment_id = Column(Integer, primary_key=True, autoincrement=True)

    # ---- NEW: link assignment to a specific admission (stay) ----
    admission_id = Column(
        Integer,
        ForeignKey("admissions.admission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Keep patient_id for convenience / fast access (denormalized)
    patient_id = Column(
        Integer,
        ForeignKey("patients.patient_id"),
        nullable=False,
        index=True,
    )

    room_id = Column(
        Integer,
        ForeignKey("rooms.room_id"),
        nullable=False,
        index=True,
    )

    start_time = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    end_time = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # ---- Relationships ----
    admission = relationship(
        "Admission",
        back_populates="room_assignments",
    )

    patient = relationship(
        "Patient",
        back_populates="assignments",
    )

    room = relationship(
        "Room",
        back_populates="assignments",
    )

    __table_args__ = (
        Index("ix_room_assignment_room_time", "room_id", "start_time", "end_time"),
        Index("ix_room_assignment_admission_time", "admission_id", "start_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<RoomAssignment("
            f"assignment_id={self.assignment_id}, "
            f"admission_id={self.admission_id}, "
            f"patient_id={self.patient_id}, "
            f"room_id={self.room_id}"
            f")>"
        )