from datetime import datetime, timezone

# ---------------- PHYSICAL CONSTANTS ----------------
# how powerful HVAC is
HEAT_POWER = 0.045       # °C per second at full power
COOL_POWER = 0.040

# natural leakage to environment
THERMAL_LEAK = 0.0025    # passive heat exchange

# ventilation mixing strength
AIRFLOW_MIX = 0.015

# stabilization near equilibrium
EQUILIBRIUM_LOCK = 0.0008


def _seconds(now, last):
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    dt = (now - last).total_seconds()
    return min(max(dt, 0.1), 5.0)


def step_temperature(
    now: datetime,
    last_time: datetime,
    current_temp: float,
    ble_temp: float,
    target_temp: float,
    mode: str,
) -> float:

    dt = _seconds(now, last_time)

    error = target_temp - current_temp

    # ---------------- HVAC ----------------
    if mode == "heat":
        hvac = HEAT_POWER * error * dt

    elif mode == "cool":
        hvac = COOL_POWER * error * dt

    elif mode == "airflow":
        hvac = AIRFLOW_MIX * (ble_temp - current_temp) * dt

    else:  # OFF
        # VERY small stabilization toward equilibrium
        hvac = EQUILIBRIUM_LOCK * (ble_temp - current_temp) * dt

    # ---------------- PASSIVE ROOM PHYSICS ----------------
    passive = THERMAL_LEAK * (ble_temp - current_temp) * dt

    new_temp = current_temp + hvac + passive

    return new_temp
