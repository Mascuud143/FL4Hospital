"""Provide diagnosis, symptom, medication, and ethnicity source data."""

from __future__ import annotations

# Build diagnosis profiles.
# - Builds one diagnosis - generate_diagnosis()
# - Builds many patient diagnoses - generate_patients()
# - Stores symptoms and medication rules

import random


DIAGNOSES = {
    "Pneumonia": {"symptoms": ["Fever", "Cough", "Shortness of breath", "Chest pain"], "medications": {"Azithromycin": [9], "Ceftriaxone": [9], "Acetaminophen": [0, 6, 12, 18]}},
    "Asthma Exacerbation": {"symptoms": ["Wheezing", "Shortness of breath", "Chest tightness"], "medications": {"Albuterol": [8, 12, 16, 20], "Prednisone": [9]}},
    "COPD Exacerbation": {"symptoms": ["Dyspnea", "Increased sputum", "Wheezing"], "medications": {"Albuterol": [8, 12, 16, 20], "Ipratropium": [8, 14, 20], "Prednisone": [9]}},
    "Heart Failure": {"symptoms": ["Shortness of breath", "Edema", "Fatigue"], "medications": {"Furosemide": [8, 16], "Lisinopril": [9], "Metoprolol": [9, 21]}},
    "Hypertension": {"symptoms": ["Headache", "Dizziness"], "medications": {"Amlodipine": [9], "Losartan": [9]}},
    "Atrial Fibrillation": {"symptoms": ["Palpitations", "Fatigue", "Dizziness"], "medications": {"Metoprolol": [9, 21], "Apixaban": [9, 21]}},
    "Sepsis": {"symptoms": ["Fever", "Hypotension", "Confusion"], "medications": {"Vancomycin": [6, 18], "Piperacillin-Tazobactam": [0, 6, 12, 18], "IV Fluids": [0]}},
    "Diabetes Mellitus": {"symptoms": ["Polyuria", "Polydipsia", "Fatigue"], "medications": {"Insulin Glargine": [21], "Insulin Lispro": [8, 12, 18]}},
    "Hypoglycemia": {"symptoms": ["Shakiness", "Sweating", "Confusion"], "medications": {"Glucose": [-1], "Glucagon": [-1]}},
    "Stroke": {"symptoms": ["Facial droop", "Weakness", "Speech difficulty"], "medications": {"Aspirin": [9], "Alteplase": [-1]}},
    "Gastrointestinal Bleeding": {"symptoms": ["Melena", "Dizziness", "Weakness"], "medications": {"Pantoprazole": [8, 20], "IV Fluids": [0]}},
    "Cholecystitis": {"symptoms": ["RUQ pain", "Fever", "Nausea"], "medications": {"Ceftriaxone": [9], "Metronidazole": [6, 14, 22]}},
}

Ethnicities = [
    "English", "Irish", "Scottish", "Welsh", "German", "French", "Italian", "Spanish", "Portuguese",
    "Polish", "Ukrainian", "Russian", "Greek", "Scandinavian", "Ashkenazi Jewish", "Sephardic Jewish",
    "Yoruba", "Igbo", "Hausa", "Akan", "Amhara", "Oromo", "Somali", "Zulu", "Xhosa", "Shona", "African American",
    "Arab", "Egyptian", "Moroccan", "Algerian", "Levantine", "Gulf Arab", "Persian", "Kurdish", "Turkish", "Armenian",
    "Punjabi", "Gujarati", "Bengali", "Tamil", "Telugu", "Marathi", "Sinhalese", "Pakistani", "Nepali",
    "Han Chinese", "Korean", "Japanese", "Tibetan", "Mongolian",
    "Vietnamese", "Thai", "Khmer", "Filipino", "Malay", "Indonesian",
    "Kazakh", "Uzbek", "Turkmen", "Uyghur",
    "Navajo", "Cherokee", "Sioux", "Maya", "Quechua", "Aymara", "Mapuche",
    "Maori", "Samoan", "Tongan", "Fijian", "Hawaiian",
    "Mexican", "Puerto Rican", "Cuban", "Afro-Caribbean", "Afro-Latino", "Mestizo", "Creole",
]


def generate_diagnosis():
    diagnosis = random.choice(list(DIAGNOSES.keys()))
    symptoms = DIAGNOSES[diagnosis]["symptoms"]
    medications = DIAGNOSES[diagnosis]["medications"]
    return diagnosis, symptoms, medications


def generate_patients(n: int) -> list[dict]:
    patients = []
    for _ in range(n):
        diagnosis, symptoms, medications = generate_diagnosis()
        patients.append({"diagnosis": diagnosis, "symptoms": symptoms, "medications": medications, "ethnicity": random.choice(Ethnicities)})
    return patients


__all__ = ["DIAGNOSES", "Ethnicities", "generate_diagnosis", "generate_patients"]
