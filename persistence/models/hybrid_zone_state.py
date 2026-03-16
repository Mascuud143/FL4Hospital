from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String

from ..database import Base


class HybridZoneState(Base):
    __tablename__ = "hybrid_zone_states"

    zone_state_id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(Integer, ForeignKey("rooms.room_id", ondelete="CASCADE"), nullable=False, index=True)
    location = Column(String, nullable=False, index=True)
    virtual_temp = Column(Float, nullable=True)
    last_ble_temp = Column(Float, nullable=True)
    hvac_mode = Column(String, nullable=False, default="off")
    last_timestamp = Column(DateTime(timezone=True), nullable=True, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_hybrid_zone_state_room_location", "room_id", "location", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<HybridZoneState(zone_state_id={self.zone_state_id}, room_id={self.room_id}, "
            f"location={self.location}, virtual_temp={self.virtual_temp}, last_ble_temp={self.last_ble_temp}, "
            f"hvac_mode={self.hvac_mode}, last_timestamp={self.last_timestamp})>"
        )
