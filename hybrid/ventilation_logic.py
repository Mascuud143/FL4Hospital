# hybrid/ventilation_logic.py

DEADBAND = 3      # stability zone (°C)
AIRFLOW_BIAS = 0.2  # small drift zone


def decide_mode(current_temp: float, target: float, airflow_requested: bool) -> str:
    """
    Stable thermostat controller with hysteresis.
    Prevents rapid mode switching.
    """

    error = current_temp - target

    # ---- inside comfort zone -> STOP HVAC ----
    if abs(error) <= DEADBAND:
        return "off"

    # ---- cooling needed ----
    if error > DEADBAND:
        return "cool" if not airflow_requested else "airflow"

    # ---- heating needed ----
    if error < -DEADBAND:
        return "heat"

    return "off"
