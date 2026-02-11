from datetime import date

# -------------------------
# Simulation horizon
# -------------------------

START_DATE = date(2026, 1, 1)
DAYS = 2  # total simulated days


# -------------------------

# Population
# -------------------------

PATIENT_COUNT = 10


# -------------------------
# Comfort behavior
# -------------------------

# Maximum number of comfort changes a patient can make per day
COMFORT_MAX_CHANGES_PER_DAY = 6


# -------------------------
# Simulation timing (SIMULATED time)
# -------------------------

# Length of one simulation step (seconds of simulated time)
SIM_STEP_S = 60              # 1 simulated minute per step

# How often sensors are sampled (seconds of simulated time)
SENSOR_SAMPLE_EVERY_S = 300  # every 5 simulated minutes


# -------------------------
# Execution control
# -------------------------

# Wall-clock sleep per simulation step.
# 0.0 = run as fast as possible
WALL_SLEEP_S = 0.0


# -------------------------
# Randomness
# -------------------------

RANDOM_SEED = 42
