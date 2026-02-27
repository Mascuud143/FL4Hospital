import math
import random


def _trunc_norm(mu: float, sigma: float, lo: float, hi: float) -> float:
    # Simple rejection sampling for a truncated normal
    while True:
        x = random.gauss(mu, sigma)
        if lo <= x <= hi:
            return x


def age_height_weight__gender_generator(min_age=10, max_age=90):
    # Bias toward older ages
    bias = 2.5
    r = random.random() ** (1 / bias)
    age = int(min_age + r * (max_age - min_age))

    # Gender independent of height/weight (more realistic)
    gender = "Male" if random.random() < 0.5 else "Female"

    # Height distribution by age group + gender (cm)
    if age < 14:
        mu_h = 155 if gender == "Male" else 152
        sd_h = 9
        lo, hi = 130, 175
    elif age < 18:
        mu_h = 172 if gender == "Male" else 165
        sd_h = 7
        lo, hi = 145, 190
    elif age < 50:
        mu_h = 178 if gender == "Male" else 165
        sd_h = 7
        lo, hi = 150, 200
    else:
        # Older adults (50+), slight height loss
        loss = 0.2 * (age - 50)  # ~2 cm per decade
        mu_h = (177 if gender == "Male" else 164) - loss
        sd_h = 7
        lo, hi = 145, 195

    height = int(_trunc_norm(mu_h, sd_h, lo, hi))

    # Weight via BMI distribution (kg) -> weight = BMI * height^2
    if age < 18:
        mu_bmi, sd_bmi = (20, 3)
    elif age < 50:
        mu_bmi, sd_bmi = (24.5, 3.5)
    else:
        mu_bmi, sd_bmi = (26.5, 4)

    bmi = _trunc_norm(mu_bmi, sd_bmi, 16, 40)
    weight = int(bmi * (height / 100) ** 2)

    return {
        "age": age,
        "height": height,  # cm
        "weight": weight,  # kg
        "gender": gender,
    }
