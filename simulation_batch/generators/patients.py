import random

NAMES = [
    "John Miller", "Emma Davis", "Lucas Brown", "Sophia Anderson",
    "Ava Martinez", "Liam Thomas"
]

DIAGNOSES = [
    "Pneumonia", "Asthma", "COPD",
    "Heart Failure", "COVID-19",
    "Hypertension", "Dehydration"
]


def generate_patients(n):
    patients = []
    for _ in range(n):
        patients.append({
            "name": random.choice(NAMES),
            "diagnosis": random.choice(DIAGNOSES)
        })
    return patients
