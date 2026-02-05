import random


def generate_comfort():
    base = random.choice([21, 22, 23])
    return {
        "temperature": base - 1.0,
    }
