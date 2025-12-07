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
from .classifier import (
    Classifier,
    FingerprintClassifier,
    MLClassifier,
    create_classifier,
    load_chirp_ml_model,
    classify_clip_ml,
)
from .detector import EventDetector
from .repository import EventRepository, SegmentRepository
from .features import (
    load_mono_wav,
    compute_avg_spectrum,
    create_mel_filterbank,
    dct,
    extract_mfcc_features,
    extract_additional_features,
    compute_spectral_features,
    compute_temporal_features,
    INT16_FULL_SCALE,
)
from .email import get_email_config, send_email
from .reporting import (
    load_events,
    filter_recent_events,
    generate_email_report,
    add_date_column,
    choose_latest_date,
    generate_chirp_report,
)

__all__ = [
    # Audio
    'AudioCapture',
    'AudioChunk',
    # Baseline
    'BaselineTracker',
    # Classifier
    'Classifier',
    'FingerprintClassifier',
    'MLClassifier',
    'create_classifier',
    'load_chirp_ml_model',
    'classify_clip_ml',
    # Detector
    'EventDetector',
    # Repository
    'EventRepository',
    'SegmentRepository',
    # Features
    'load_mono_wav',
    'compute_avg_spectrum',
    'create_mel_filterbank',
    'dct',
    'extract_mfcc_features',
    'extract_additional_features',
    'compute_spectral_features',
    'compute_temporal_features',
    'INT16_FULL_SCALE',
    # Email
    'get_email_config',
    'send_email',
    # Reporting
    'load_events',
    'filter_recent_events',
    'generate_email_report',
    'add_date_column',
    'choose_latest_date',
    'generate_chirp_report',
]

