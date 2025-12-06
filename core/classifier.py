"""
Classifier interface and implementations.

Open/Closed Principle: Open for extension (new classifier types),
closed for modification (existing code doesn't change).
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

# Import existing classification functions
import monitor


class Classifier(ABC):
    """
    Abstract base class for event classifiers.
    
    Interface Segregation: Small, focused interface.
    Dependency Inversion: Code depends on this abstraction.
    """
    
    @abstractmethod
    def classify(
        self,
        event_chunks: List[bytes],
        duration_sec: float,
        config: dict
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """
        Classify an event as chirp or not.
        
        Args:
            event_chunks: List of raw PCM byte chunks
            duration_sec: Event duration in seconds
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_chirp, similarity/confidence, confidence_score, rejection_reason)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if classifier is available/loaded."""
        pass


class FingerprintClassifier(Classifier):
    """
    Spectral fingerprint-based classifier.
    
    Single Responsibility: Fingerprint-based classification.
    Liskov Substitution: Can be used anywhere Classifier is expected.
    """
    
    def __init__(self, config: dict):
        """Initialize fingerprint classifier."""
        self.config = config
        self._fingerprint_info = None
        self._load_fingerprint()
    
    def _load_fingerprint(self) -> None:
        """Load fingerprint from file."""
        fingerprint_file = Path(self.config["chirp_classification"]["fingerprint_file"])
        
        if not fingerprint_file.exists():
            return
        
        try:
            import json
            data = json.load(fingerprint_file.open())
            fp = np.array(data["fingerprint"], dtype=np.float32)
            fp = fp / (np.linalg.norm(fp) + 1e-9)
            
            self._fingerprint_info = {
                "fingerprint": fp,
                "sample_rate": data["sample_rate"],
                "fft_size": data["fft_size"]
            }
        except Exception:
            pass
    
    def is_available(self) -> bool:
        """Check if fingerprint is loaded."""
        return self._fingerprint_info is not None
    
    def classify(
        self,
        event_chunks: List[bytes],
        duration_sec: float,
        config: dict
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """Classify using fingerprint method."""
        if not self.is_available():
            return False, None, None, "no_fingerprint"
        
        return monitor.classify_event_is_chirp(
            event_chunks,
            self._fingerprint_info,
            duration_sec,
            config
        )


class MLClassifier(Classifier):
    """
    Machine learning-based classifier.
    
    Single Responsibility: ML-based classification.
    Liskov Substitution: Can be used anywhere Classifier is expected.
    """
    
    def __init__(self, config: dict):
        """Initialize ML classifier."""
        self.config = config
        self._model_info = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ML model from files."""
        try:
            from scripts.classify_chirp_ml import load_ml_model
            self._model_info = load_ml_model(self.config)
        except Exception:
            pass
    
    def is_available(self) -> bool:
        """Check if model is loaded."""
        return self._model_info is not None
    
    def classify(
        self,
        event_chunks: List[bytes],
        duration_sec: float,
        config: dict
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """Classify using ML model."""
        if not self.is_available():
            return False, None, None, "no_model"
        
        return monitor.classify_event_is_chirp_ml(
            event_chunks,
            self._model_info,
            duration_sec,
            config
        )


def create_classifier(config: dict) -> Optional[Classifier]:
    """
    Factory function to create appropriate classifier.
    
    Dependency Inversion: Returns abstraction, not concrete class.
    """
    chirp_cfg = config["chirp_classification"]
    use_ml = chirp_cfg.get("use_ml_classifier", False)
    
    if use_ml:
        classifier = MLClassifier(config)
        if classifier.is_available():
            return classifier
        # Fall back to fingerprint
        print("[WARN] ML model not found, falling back to fingerprint")
    
    classifier = FingerprintClassifier(config)
    if classifier.is_available():
        return classifier
    
    return None

