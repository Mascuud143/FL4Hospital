# hybrid/ventilation_logic.py

DEADBAND = 0.5
HYSTERESIS = 0.2


def decide_mode(current_temp: float, target: float, airflow_requested: bool, current_mode: str = "off") -> str:
    """
    Stable thermostat controller with hysteresis.
    Prevents rapid mode switching.
    """

    error = current_temp - target

    stop_threshold = DEADBAND
    start_threshold = DEADBAND + HYSTERESIS

    if current_mode in {"heat", "cool", "airflow"} and abs(error) <= start_threshold:
        return "off"

    if abs(error) <= stop_threshold:
        return "off"

    if error > start_threshold:
        return "cool" if not airflow_requested else "airflow"

    if error < -start_threshold:
        return "heat"

    return current_mode if current_mode in {"heat", "cool", "airflow"} else "off"
