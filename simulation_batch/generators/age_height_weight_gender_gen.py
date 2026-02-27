import random

def age_height_weight__gender_generator(min_age=10, max_age=90):
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
    

    # decide gender based on height and weight (not perfect, but adds some correlation)
    # lets have probability of being male increase with height and weight
    p_male = 0.5
    if height > 180:
        p_male += 0.2
    if weight > 80:
        p_male += 0.1
    p_male = min(0.9, p_male)  # cap at 90%

    gender = "Male" if random.random() < p_male else "Female"
    

    return {
        "age": age,
        "height": height,  # cm
        "weight": weight,  # kg
        "gender": gender
    }
