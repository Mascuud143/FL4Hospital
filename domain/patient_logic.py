from dataclasses import dataclass


@dataclass
class Patient:
    id: int
    is_active: bool = True
    is_discharged: bool = False


def can_patient_be_assigned(patient: Patient) -> bool:
    """
    Determines whether a patient is eligible for room assignment.
    """
    if not patient.is_active:
        return False

    if patient.is_discharged:
        return False

    return True
