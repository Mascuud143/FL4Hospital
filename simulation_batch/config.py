from datetime import date

# -------------------------
# Simulation horizon
# -------------------------

START_DATE = date(2026, 1, 1)
DAYS = 365*1 # total simulated days


# -------------------------
# Population
# -------------------------

PATIENT_COUNT = 100


# -------------------------
# Room transfer behavior
# -------------------------

# Probability that a patient changes rooms once during their stay
# 0.0 = never, 1.0 = always
CHANGE_ROOM_PROB = 0.3

# Minimum days after admission before a transfer is allowed
MIN_DAYS_BEFORE_TRANSFER = 1

# Minimum days before discharge after a transfer
MIN_DAYS_AFTER_TRANSFER = 1


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
# Generation toggles
# -------------------------

ENABLE_COMFORT = True
ENABLE_MEDICATION = True
ENABLE_VISITS = True

ENABLE_TOILET_USAGE = True
ENABLE_SENSOR_EMIT = True
ENABLE_UTILITY_USAGE = True


# -------------------------
# Randomness
# -------------------------

RANDOM_SEED = 42
