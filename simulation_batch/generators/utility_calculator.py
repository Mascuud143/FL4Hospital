def calculate_energy_usage_kwh(
    power_watts: float,
    duration_hours: float,
) -> float:
    """
    Convert power usage to kWh.
    """
    return (power_watts * duration_hours) / 1000


def calculate_daily_energy_cost(
    energy_kwh: float,
    cost_per_kwh: float,
) -> float:
    """
    Calculate cost based on energy consumption.
    """
    return energy_kwh * cost_per_kwh
