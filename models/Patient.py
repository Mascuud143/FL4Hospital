class Patient:
    def __init__(self, patient_id, age, gender,height, ethnicty, weight,admission_date, room_id, current_diagnosis, other_diagnoses):
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.height = height
        self.ethnicty = ethnicty
        self.weight = weight
        self.admission_date = admission_date
        self.room_id = room_id
        self.current_diagnosis = current_diagnosis
        self.other_diagnoses = other_diagnoses