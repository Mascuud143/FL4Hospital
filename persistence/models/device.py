from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class Device(Base):
    __tablename__ = "devices"

    device_id = Column(Integer, primary_key=True, autoincrement=True)
    mac_address = Column(String, nullable=False, unique=True, index=True)
    device_type = Column(String, nullable=False)
    room_id = Column(Integer, ForeignKey("rooms.room_id"), nullable=True)

    # Relationships
    room = relationship("Room", back_populates="devices")
    sensors = relationship("Sensor", back_populates="device", cascade="all, delete-orphan")

    ventilation = relationship("Ventilation", back_populates="device", uselist=False, cascade="all, delete-orphan")
    speaker = relationship("Speaker", back_populates="device", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Device(device_id={self.device_id}, mac_address={self.mac_address}, room_id={self.room_id})>"
