from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base


class ToiletLight(Base):
    __tablename__ = "toilet_lights"

    toilet_light_id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.device_id"), nullable=False, unique=True)

    state = Column(Boolean, nullable=False)  # True = ON, False = OFF
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    device = relationship("Device", back_populates="toilet_light")

    def __repr__(self) -> str:
        return (
            f"<ToiletLight(toilet_light_id={self.toilet_light_id}, "
            f"device_id={self.device_id}, state={self.state}, "
            f"timestamp={self.timestamp})>"
        )
