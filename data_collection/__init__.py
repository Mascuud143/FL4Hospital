"""
Data Collection Layer (buffer-free)

Receives sensor/actuator events (e.g., from BLEManager), validates/normalizes them,
then forwards clean events to downstream sinks (DB, CSV, dashboard, etc).
"""

from .data_collector import DataCollector
from .data_processor import DataProcessor

__all__ = ["DataCollector", "DataProcessor"]
