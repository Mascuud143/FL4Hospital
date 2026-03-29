import random


def generate_comfort():
    base = random.choice([21, 22, 23])
    return {
        "temperature": round(random.uniform(base - 1.0, base + 1.0), 1),
        # "humidity": random.uniform(40.0, 60.0),
        "light_intensity": random.uniform(0, 100),
        "sound_level": random.uniform(0, 100),
        "ventilation": random.uniform(0, 100),
        "source": "Simulation"
    }
