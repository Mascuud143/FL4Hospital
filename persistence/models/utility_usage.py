from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class UtilityUsage(Base):
    __tablename__ = "utility_usages"

    usage_id = Column(Integer, primary_key=True, autoincrement=True)

    category = Column(
        String,
        nullable=False,
        doc="hvac | toilet_heater | toilet_light | water",
        index=True,
    )

    water_consumption = Column(Float, nullable=True)  # liters
    power_consumption = Column(Float, nullable=True)  # kWh

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

    room_id = Column(
        Integer,
        ForeignKey("rooms.room_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ✅ NEW: which device caused this usage (nullable for water unless you have a meter device)
    device_id = Column(
        Integer,
        ForeignKey("devices.device_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    room = relationship("Room", back_populates="utility_usages")
    device = relationship("Device")

    __table_args__ = (
        Index("ix_utility_usage_room_category_time", "room_id", "category", "start_time"),
        Index("ix_utility_usage_room_time", "room_id", "start_time"),
        Index("ix_utility_usage_device_time", "device_id", "start_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<UtilityUsage("
            f"id={self.usage_id}, "
            f"room_id={self.room_id}, "
            f"device_id={self.device_id}, "
            f"category={self.category}, "
            f"power={self.power_consumption}, "
            f"water={self.water_consumption}, "
            f"start={self.start_time}, "
            f"end={self.end_time}"
            f")>"
        )
