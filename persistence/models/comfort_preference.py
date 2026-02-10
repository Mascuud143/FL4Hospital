from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Boolean,
    String,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class ComfortPreference(Base):
    __tablename__ = "comfort_preferences"

    comfort_pref_id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Comfort targets (human intent) ---
    temperature_main = Column(Float, nullable=False)
    temperature_toilet = Column(Float, nullable=True)

    light_intensity = Column(Float, nullable=True)
    sound_level = Column(Float, nullable=True)

    # Request for better air circulation (NOT heating/cooling)
    airflow = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="If True, system should enable ventilation airflow mode only",
    )

    # --- Ownership ---
    patient_id = Column(
        Integer,
        ForeignKey("patients.patient_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    room_id = Column(
        Integer,
        ForeignKey("rooms.room_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # --- Metadata ---
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    source = Column(
        String,
        nullable=False,
        doc="simulation | manual | ai",
    )

    # --- Relationships ---
    patient = relationship(
        "Patient",
        back_populates="comfort_preferences",
    )

    room = relationship(
        "Room",
        back_populates="comfort_preferences",
    )

    __table_args__ = (
        Index("ix_comfort_pref_room_time", "room_id", "timestamp"),
        Index("ix_comfort_pref_patient_time", "patient_id", "timestamp"),
        Index("ix_comfort_pref_room_airflow", "room_id", "airflow"),
    )

    def __repr__(self) -> str:
        return (
            f"<ComfortPreference("
            f"id={self.comfort_pref_id}, "
            f"room_id={self.room_id}, "
            f"patient_id={self.patient_id}, "
            f"temp_main={self.temperature_main}, "
            f"temp_toilet={self.temperature_toilet}, "
            f"airflow={self.airflow}, "
            f"source={self.source}, "
            f"timestamp={self.timestamp}"
            f")>"
        )
