# Import all models so Base.metadata knows them when init_db() runs.

from .room import Room
from .device import Device
from .sensor import Sensor
from .data import Data

from .patient import Patient
from .comfort_preference import ComfortPreference
from .room_assignment import RoomAssignment
from .utility_usage import UtilityUsage

from .ventilation import Ventilation
from .speaker import Speaker
from .toilet_light import ToiletLight
from .toilet_heater import ToiletHeater

__all__ = [
"Room",
"Device",
"Sensor",
"Data",
"Patient",
"ComfortPreference",
"RoomAssignment",
"UtilityUsage",
"Ventilation",
"Speaker",
"ToiletLight",
"ToiletHeater",
]
