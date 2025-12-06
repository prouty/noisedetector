"""
Core domain models and interfaces for the noise detector system.

This module follows SOLID principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Open for extension via interfaces, closed for modification
- Liskov Substitution: Implementations are interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

from .audio import AudioCapture, AudioChunk
from .baseline import BaselineTracker
from .classifier import Classifier, FingerprintClassifier, MLClassifier, create_classifier
from .detector import EventDetector
from .repository import EventRepository, SegmentRepository

__all__ = [
    'AudioCapture',
    'AudioChunk',
    'BaselineTracker',
    'Classifier',
    'FingerprintClassifier',
    'MLClassifier',
    'create_classifier',
    'EventDetector',
    'EventRepository',
    'SegmentRepository',
]

