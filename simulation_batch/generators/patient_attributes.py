from __future__ import annotations

# Build patient base attributes.
# - Creates age, gender, height, and weight
# - Uses truncated normal values - _trunc_norm()
# - Builds one patient base record - age_height_weight__gender_generator()

import random


def _trunc_norm(mu: float, sigma: float, lo: float, hi: float) -> float:
    while True:
        value = random.gauss(mu, sigma)
        if lo <= value <= hi:
            return value


def age_height_weight__gender_generator(min_age: int = 10, max_age: int = 90) -> dict[str, int | str]:
    bias = 2.5
    age = int(min_age + (random.random() ** (1 / bias)) * (max_age - min_age))
    gender = "Male" if random.random() < 0.5 else "Female"
    if age < 14:
        mu_h, sd_h, lo, hi = (155, 9, 130, 175) if gender == "Male" else (152, 9, 130, 175)
    elif age < 18:
        mu_h, sd_h, lo, hi = (172, 7, 145, 190) if gender == "Male" else (165, 7, 145, 190)
    elif age < 50:
        mu_h, sd_h, lo, hi = (178, 7, 150, 200) if gender == "Male" else (165, 7, 150, 200)
    else:
        loss = 0.2 * (age - 50)
        mu_h, sd_h, lo, hi = ((177 - loss), 7, 145, 195) if gender == "Male" else ((164 - loss), 7, 145, 195)
    height = int(_trunc_norm(mu_h, sd_h, lo, hi))
    if age < 18:
        mu_bmi, sd_bmi = 20, 3
    elif age < 50:
        mu_bmi, sd_bmi = 24.5, 3.5
    else:
        mu_bmi, sd_bmi = 26.5, 4
    bmi = _trunc_norm(mu_bmi, sd_bmi, 16, 40)
    weight = int(bmi * (height / 100) ** 2)
    return {"age": age, "height": height, "weight": weight, "gender": gender}


__all__ = ["age_height_weight__gender_generator"]
