import random

def age_height_weight_generator(min_age=10, max_age=90):
    # Bias toward older ages
    bias = 2.5
    r = random.random() ** (1 / bias)
    age = int(min_age + r * (max_age - min_age))

    if age < 14:
        # Children (10–13)
        height = random.randint(135, 165)
        weight = random.randint(30, 55)

    elif age < 18:
        # Teens (14–17)
        height = random.randint(150, 180)
        weight = random.randint(45, 75)

    elif age < 50:
        # Adults
        height = random.randint(155, 195)
        weight = random.randint(55, 110)

    else:
        # Older adults (50+), slight height loss
        height = random.randint(150, 185) - max(0, (age - 50) // 10)
        weight = random.randint(50, 100)

    return {
        "age": age,
        "height": height,  # cm
        "weight": weight   # kg
    }
