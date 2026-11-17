"""Participant-level assessment of EEG reference dependence."""

from .assessment import AssessmentResult, NoCommonComponentError, assess_reference_dependence
from .spectral import BandpowerClassifier

__all__ = [
    "AssessmentResult",
    "BandpowerClassifier",
    "NoCommonComponentError",
    "assess_reference_dependence",
]
