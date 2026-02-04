from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class Data(Base):
    __tablename__ = "data"

    data_id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(Integer, ForeignKey("sensors.sensor_id"), nullable=False, index=True)

    value = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    sensor = relationship("Sensor", back_populates="data_points")

    __table_args__ = (
        Index("ix_data_sensor_time", "sensor_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Data(data_id={self.data_id}, sensor_id={self.sensor_id}, value={self.value}, timestamp={self.timestamp})>"
