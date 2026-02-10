from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base


class ToiletHeater(Base):
    __tablename__ = "toilet_heaters"

    toilet_heater_id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.device_id"), nullable=False, unique=True)

    state = Column(Boolean, nullable=False)  # True = ON, False = OFF
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    device = relationship("Device", back_populates="toilet_heater")

    def __repr__(self) -> str:
        return (
            f"<ToiletHeater(toilet_heater_id={self.toilet_heater_id}, "
            f"device_id={self.device_id}, state={self.state}, "
            f"timestamp={self.timestamp})>"
        )
