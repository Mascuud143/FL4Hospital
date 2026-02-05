from dataclasses import dataclass


@dataclass
class ComfortPreference:
    min_temp: float
    max_temp: float


def is_temperature_comfortable(
    temperature: float,
    preference: ComfortPreference
) -> bool:
    """
    Checks if the current temperature is within comfort range.
    """
    return preference.min_temp <= temperature <= preference.max_temp


def comfort_action(
    temperature: float,
    preference: ComfortPreference
) -> str:
    """
    Decide what action the system should take.
    """
    if temperature < preference.min_temp:
        return "INCREASE_HEATING"

    if temperature > preference.max_temp:
        return "INCREASE_COOLING"

    return "NO_ACTION"
