import random


DIAGNOSES = [
    # Respiratory
    "Pneumonia",
    "Asthma Exacerbation",
    "COPD Exacerbation",
    "Acute Respiratory Failure",

    # Cardiovascular
    "Heart Failure",
    "Hypertension",
    "Coronary Artery Disease",
    "Atrial Fibrillation",
    "Chest Pain / Suspected Myocardial Infarction",
    "Stroke",

    # Infections
    "Sepsis",
    "Urinary Tract Infection",
    "Cellulitis",

    # Metabolic / Renal
    "Dehydration",
    "Acute Kidney Injury",
    "Diabetes Mellitus",
    "Hypoglycemia",
    "Electrolyte Imbalance",

    # Neurological / Falls
    "Syncope",
    "Fall With Injury",
    "Hip Fracture",
    "Delirium",

    # Gastrointestinal
    "Gastrointestinal Bleeding",
    "Abdominal Pain",
    "Cholecystitis"
]
Ethnicities = [
    # European
    "English", "Irish", "Scottish", "Welsh",
    "German", "French", "Italian", "Spanish", "Portuguese",
    "Polish", "Ukrainian", "Russian", "Greek",
    "Scandinavian", "Ashkenazi Jewish", "Sephardic Jewish",

    # Sub-Saharan African
    "Yoruba", "Igbo", "Hausa", "Akan",
    "Amhara", "Oromo", "Somali",
    "Zulu", "Xhosa", "Shona",
    "African American",

    # North African / Middle Eastern
    "Arab", "Egyptian", "Moroccan", "Algerian",
    "Levantine", "Gulf Arab",
    "Persian", "Kurdish", "Turkish", "Armenian",

    # South Asian
    "Punjabi", "Gujarati", "Bengali", "Tamil",
    "Telugu", "Marathi", "Sinhalese",
    "Pakistani", "Nepali",

    # East Asian
    "Han Chinese", "Korean", "Japanese",
    "Tibetan", "Mongolian",

    # Southeast Asian
    "Vietnamese", "Thai", "Khmer",
    "Filipino", "Malay", "Indonesian",

    # Central Asian
    "Kazakh", "Uzbek", "Turkmen", "Uyghur",

    # Indigenous Peoples (Americas)
    "Navajo", "Cherokee", "Sioux",
    "Maya", "Quechua", "Aymara", "Mapuche",

    # Pacific Islander
    "Maori", "Samoan", "Tongan",
    "Fijian", "Hawaiian",

    # Latin American / Mixed
    "Mexican", "Puerto Rican", "Cuban",
    "Afro-Caribbean", "Afro-Latino",
    "Mestizo", "Creole",
]



def generate_patients(n):
    patients = []
    for _ in range(n):
        patients.append({
            "diagnosis": random.choice(DIAGNOSES),
            "ethnicity": random.choice(Ethnicities)
        })
    return patients
