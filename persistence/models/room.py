from __future__ import annotations

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class Room(Base):
    __tablename__ = "rooms"

    room_id = Column(Integer, primary_key=True, autoincrement=True)
    room_number = Column(String, nullable=False, unique=True)

    # Relationships
    devices = relationship("Device", back_populates="room", cascade="all, delete-orphan")
    utility_usages = relationship("UtilityUsage", back_populates="room", cascade="all, delete-orphan")
    assignments = relationship("RoomAssignment", back_populates="room", cascade="all, delete-orphan")
    comfort_preferences = relationship("ComfortPreference", back_populates="room", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Room(room_id={self.room_id}, room_number={self.room_number})>"
