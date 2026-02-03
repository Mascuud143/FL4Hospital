class Preference:
    def __init__(self, preference_id, patient_id, humidity_level, temperature_level, light_intensity, simulation_id):
        self.preference_id = preference_id  
        self.light_intensity = light_intensity
        self.humidity_level = humidity_level
        self.temperature_level = temperature_level
        self.patient_id = patient_id
        self.simulation_id = simulation_id