from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class Sensor(Base):
    __tablename__ = "sensors"

    sensor_id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("devices.device_id"), nullable=False, index=True)

    unit = Column(String, nullable=True)
    uuid = Column(String, nullable=False, index=True)
    sensor_type = Column(String, nullable=False, index=True)

    # Relationships
    device = relationship("Device", back_populates="sensors")
    data_points = relationship("Data", back_populates="sensor", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Sensor(sensor_id={self.sensor_id}, device_id={self.device_id}, type={self.sensor_type}, uuid={self.uuid})>"
