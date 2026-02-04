from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class UtilityUsage(Base):
    __tablename__ = "utility_usage"

    usage_id = Column(Integer, primary_key=True, autoincrement=True)

    water_consumption = Column(Float, nullable=True)
    power_consumption = Column(Float, nullable=True)

    start_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)

    room_id = Column(Integer, ForeignKey("rooms.room_id"), nullable=False, index=True)

    # Relationships
    room = relationship("Room", back_populates="utility_usages")

    def __repr__(self) -> str:
        return f"<UtilityUsage(usage_id={self.usage_id}, room_id={self.room_id})>"
