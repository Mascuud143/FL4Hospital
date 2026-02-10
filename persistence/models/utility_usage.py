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

    # What caused this usage session
    category = Column(
        String,
        nullable=False,
        doc="hvac | toilet_heater | toilet_light | water",
        index=True,
    )

    # Consumption accumulated over [start_time, end_time]
    water_consumption = Column(Float, nullable=True)  # liters (or your chosen unit)
    power_consumption = Column(Float, nullable=True)  # kWh (recommended)

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

    # Relationships
    room = relationship("Room", back_populates="utility_usages")

    __table_args__ = (
        Index("ix_utility_usage_room_category_time", "room_id", "category", "start_time"),
        Index("ix_utility_usage_room_time", "room_id", "start_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<UtilityUsage("
            f"id={self.usage_id}, "
            f"room_id={self.room_id}, "
            f"category={self.category}, "
            f"power={self.power_consumption}, "
            f"water={self.water_consumption}, "
            f"start={self.start_time}, "
            f"end={self.end_time}"
            f")>"
        )
