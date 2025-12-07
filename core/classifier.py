"""
Classifier interface and implementations.

Open/Closed Principle: Open for extension (new classifier types),
closed for modification (existing code doesn't change).

Single Responsibility: All classification logic is contained here.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import json

from core.audio import AudioCapture

# Constants for audio processing
INT16_FULL_SCALE = AudioCapture.INT16_FULL_SCALE


# ============================================================================
# Classification implementation functions
# ============================================================================

def load_chirp_fingerprint(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load fingerprint from file."""
    fingerprint_file = Path(config["chirp_classification"]["fingerprint_file"])
    if not fingerprint_file.exists():
        print("[INFO] No chirp_fingerprint.json found; chirp classification disabled.")
        return None

    try:
        data = json.load(fingerprint_file.open())
        fp = np.array(data["fingerprint"], dtype=np.float32)
        fp = fp / (np.linalg.norm(fp) + 1e-9)
        sr = data["sample_rate"]
        fft_size = data["fft_size"]
        
        # Validate sample rate matches current configuration
        expected_sr = config["audio"]["sample_rate"]
        if sr != expected_sr:
            print(f"[WARN] Fingerprint sample rate ({sr} Hz) doesn't match config sample_rate ({expected_sr} Hz). Chirp classification may be inaccurate.")
        
        print(f"[INFO] Loaded chirp fingerprint from {fingerprint_file} (sr={sr}Hz, fft_size={fft_size})")
        return {"fingerprint": fp, "sample_rate": sr, "fft_size": fft_size}
    except Exception as e:
        print(f"[WARN] Failed to load chirp fingerprint: {e}")
        return None


def load_chirp_ml_model(config: Dict[str, Any]) -> Optional[Tuple]:
    """
    Load ML model for chirp classification.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, scaler, metadata) or None if not found
    """
    try:
        import joblib
        
        metadata_file = Path(config["chirp_classification"].get("ml_metadata_file", "data/chirp_model_metadata.json"))
        
        if not metadata_file.exists():
            return None
        
        try:
            with metadata_file.open() as f:
                metadata = json.load(f)
            
            model_file = metadata_file.parent / metadata["model_file"]
            scaler_file = metadata_file.parent / metadata["scaler_file"]
            
            if not model_file.exists() or not scaler_file.exists():
                return None
            
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            
            print(f"[INFO] Loaded ML model: {metadata.get('model_type', 'unknown')} "
                  f"(train_acc={metadata.get('metrics', {}).get('train_accuracy', 0):.3f})")
            return model, scaler, metadata
        except Exception as e:
            print(f"[WARN] Failed to load ML model: {e}")
            return None
    except ImportError:
        print("[WARN] scikit-learn not available - ML classification disabled")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load ML model: {e}")
        return None


def compute_event_spectrum_from_chunks(
    chunks: List[bytes], 
    sample_rate: int, 
    fft_size: int
) -> Optional[np.ndarray]:
    """
    Compute averaged magnitude spectrum from event audio chunks.
    
    This function:
    1. Concatenates all chunks into single audio stream
    2. Removes DC offset (hardware artifact)
    3. Applies Hanning window and FFT with 50% overlap
    4. Averages multiple FFT windows for stability
    5. Normalizes result (L2 norm) for cosine similarity
    
    Args:
        chunks: List of raw PCM byte chunks (int16 little-endian)
        sample_rate: Audio sample rate in Hz
        fft_size: FFT window size (must match fingerprint fft_size)
        
    Returns:
        Normalized magnitude spectrum (1D numpy array) or None if computation fails
    """
    if not chunks:
        return None

    try:
        raw = b"".join(chunks)
        if len(raw) == 0:
            return None
        
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        
        # Remove DC offset before spectral analysis
        dc_offset = np.mean(samples)
        samples = samples - dc_offset

        # Zero-pad if audio is shorter than FFT window
        if samples.shape[0] < fft_size:
            pad = fft_size - samples.shape[0]
            samples = np.pad(samples, (0, pad), mode='constant')
    except Exception as e:
        return None

    # FFT parameters
    hop = fft_size // 2  # 50% overlap
    window = np.hanning(fft_size)
    specs = []

    # Compute FFT windows with overlap
    max_start = max(0, len(samples) - fft_size)
    
    if max_start == 0 and len(samples) >= fft_size:
        # Single window case
        chunk = samples[0:fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)
    else:
        # Multiple windows with 50% overlap
        for start in range(0, max_start, hop):
            chunk = samples[start:start + fft_size] * window
            spec = np.abs(np.fft.rfft(chunk))
            specs.append(spec)
    
    if not specs:
        return None
    
    # Apply high-pass filter in frequency domain
    freq_resolution = sample_rate / fft_size
    cutoff_bin = max(1, int(20.0 / freq_resolution))  # 20 Hz high-pass
    
    for i, spec in enumerate(specs):
        spec[:cutoff_bin] = 0.0
        specs[i] = spec

    # Average all FFT windows for stability
    avg_spec = np.mean(specs, axis=0)
    
    # Normalize to unit length (L2 norm)
    norm = np.linalg.norm(avg_spec)
    if norm < 1e-9:
        return None
    
    avg_spec = avg_spec / norm
    return avg_spec


def compute_attack_decay_ratio(chunk_rms_values: list) -> Optional[float]:
    """
    Calculate attack/decay ratio. Higher values indicate sharp attack (chirp-like).
    Returns ratio of attack time to decay time, or None if insufficient data.
    """
    if len(chunk_rms_values) < 3:
        return None
    
    # Find peak
    peak_idx = chunk_rms_values.index(max(chunk_rms_values))
    if peak_idx == 0 or peak_idx == len(chunk_rms_values) - 1:
        return None
    
    # Attack: time to reach 90% of peak from start
    peak_value = chunk_rms_values[peak_idx]
    attack_threshold = peak_value * 0.9
    attack_time = peak_idx
    for i in range(peak_idx):
        if chunk_rms_values[i] >= attack_threshold:
            attack_time = peak_idx - i
            break
    
    # Decay: time to drop to 10% of peak after peak
    decay_threshold = peak_value * 0.1
    decay_time = len(chunk_rms_values) - peak_idx - 1
    for i in range(peak_idx + 1, len(chunk_rms_values)):
        if chunk_rms_values[i] <= decay_threshold:
            decay_time = i - peak_idx
            break
    
    if decay_time == 0:
        return None
    
    return attack_time / decay_time


def compute_spectral_centroid(spectrum: np.ndarray, sample_rate: int, fft_size: int) -> float:
    """
    Calculate spectral centroid (weighted frequency center of mass).
    Higher values indicate more high-frequency content (chirp-like).
    """
    freq_resolution = sample_rate / fft_size
    frequencies = np.arange(len(spectrum)) * freq_resolution
    
    magnitude = np.abs(spectrum)
    total_magnitude = np.sum(magnitude)
    
    if total_magnitude == 0:
        return 0.0
    
    centroid = np.sum(frequencies * magnitude) / total_magnitude
    return float(centroid)


def find_best_chirp_segment(
    event_chunks: List[bytes],
    fingerprint_info: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[Optional[List[bytes]], Optional[float], Optional[str]]:
    """
    Find the best segment within event_chunks that matches the chirp fingerprint.
    Uses sliding windows to find the segment with highest similarity and best frequency characteristics.
    
    Returns:
        (best_chunks, best_similarity, rejection_reason) or (None, None, reason) if no good segment found
    """
    if fingerprint_info is None or not event_chunks:
        return None, None, "no_fingerprint_or_chunks"
    
    chirp_cfg = config["chirp_classification"]
    freq_cfg = chirp_cfg["frequency_filtering"]
    fp = fingerprint_info["fingerprint"]
    fft_size = fingerprint_info["fft_size"]
    sr = fingerprint_info["sample_rate"]
    
    # Try different window sizes and positions
    num_chunks = len(event_chunks)
    if num_chunks < 1:
        return None, None, "insufficient_chunks"
    
    window_sizes = [0.14, 0.25, 0.33, 0.5, 0.75, 1.0]
    best_chunks = None
    best_similarity = -1.0
    best_score = -1.0
    
    freq_resolution = sr / fft_size
    fan_noise_max_bin = int(freq_cfg["fan_noise_max_freq_hz"] / freq_resolution)
    chirp_min_bin = int(freq_cfg["chirp_min_freq_hz"] / freq_resolution)
    
    for window_size in window_sizes:
        num_chunks_in_window = max(1, int(num_chunks * window_size))
        start_idx = num_chunks - num_chunks_in_window
        window_chunks = event_chunks[start_idx:]
        
        if not window_chunks:
            continue
        
        # Compute spectrum for this window
        window_spec = compute_event_spectrum_from_chunks(window_chunks, sr, fft_size)
        if window_spec is None:
            continue
        
        # Calculate similarity to chirp fingerprint
        sim = float(np.dot(fp, window_spec))
        
        # If non-chirp fingerprint is available, also calculate similarity to it
        non_chirp_sim = None
        if "non_chirp_fingerprint" in fingerprint_info:
            non_chirp_fp = fingerprint_info["non_chirp_fingerprint"]
            non_chirp_sim = float(np.dot(non_chirp_fp, window_spec))
        
        # Calculate frequency characteristics
        total_energy = np.sum(window_spec)
        if total_energy == 0:
            continue
        
        low_freq_energy = np.sum(window_spec[:fan_noise_max_bin])
        high_freq_energy = np.sum(window_spec[chirp_min_bin:])
        low_freq_ratio = low_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
        
        # Score: combine similarity with frequency quality
        freq_score = 1.0
        passes_low_freq = low_freq_ratio <= freq_cfg["low_freq_energy_threshold"]
        min_high_freq = freq_cfg.get("high_freq_energy_min_ratio", 0.1)
        passes_high_freq = high_freq_ratio >= min_high_freq
        
        if passes_low_freq and passes_high_freq:
            freq_score = 1.2
        elif passes_low_freq:
            freq_score = 1.1
        elif passes_high_freq:
            freq_score = 1.05
        else:
            if low_freq_ratio > freq_cfg["low_freq_energy_threshold"]:
                freq_score *= (1.0 - (low_freq_ratio - freq_cfg["low_freq_energy_threshold"]))
            if high_freq_ratio < min_high_freq:
                freq_score *= (high_freq_ratio / min_high_freq)
        
        # Combined score
        window_size_factor = 1.0 + (1.0 - window_size) * 0.1
        
        non_chirp_penalty = 1.0
        if non_chirp_sim is not None:
            non_chirp_penalty = max(0.5, 1.0 - non_chirp_sim * 0.5)
        
        combined_score = sim * max(0.0, freq_score) * window_size_factor * non_chirp_penalty
        
        if combined_score > best_score:
            best_score = combined_score
            best_similarity = sim
            best_chunks = window_chunks
    
    if best_chunks is None or best_similarity < 0:
        return None, None, "no_valid_segment"
    
    return best_chunks, best_similarity, None


def classify_event_is_chirp_ml(
    event_chunks: List[bytes],
    ml_model_info: Tuple,
    duration_sec: float,
    config: Dict[str, Any]
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Classify event using ML model.
    
    Args:
        event_chunks: List of raw PCM byte chunks
        ml_model_info: Tuple from load_chirp_ml_model()
        duration_sec: Event duration
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_chirp, similarity, confidence, rejection_reason)
    """
    try:
        from core.features import extract_mfcc_features, extract_additional_features
        
        model, scaler, metadata = ml_model_info
        sr = config["audio"]["sample_rate"]
        
        # Convert chunks to samples
        samples_list = []
        for chunk in event_chunks:
            chunk_samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
            samples_list.append(chunk_samples)
        
        if not samples_list:
            return False, None, None, "no_audio_data"
        
        # Concatenate all chunks
        samples = np.concatenate(samples_list)
        
        # Extract features
        mfcc_features = extract_mfcc_features(samples, sr)
        additional_features = extract_additional_features(samples, sr)
        features = np.concatenate([mfcc_features, additional_features])
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        is_chirp = bool(prediction == 1)
        confidence = float(probability[1] if is_chirp else probability[0])
        
        # For ML, we use confidence as both similarity and confidence
        return is_chirp, confidence, confidence, None
        
    except Exception as e:
        return False, None, None, f"ml_exception_{str(e)}"


def classify_clip_ml(clip_path: Path, model_info: Tuple) -> Tuple[bool, float, Optional[str]]:
    """
    Classify a clip file using ML model.
    
    Args:
        clip_path: Path to WAV file
        model_info: Tuple from load_chirp_ml_model()
        
    Returns:
        Tuple of (is_chirp, confidence, error_message)
    """
    try:
        from core.features import load_mono_wav, extract_mfcc_features, extract_additional_features
        
        model, scaler, metadata = model_info
        
        # Load audio
        samples, sr = load_mono_wav(clip_path)
        
        # Extract features
        mfcc_features = extract_mfcc_features(samples, sr)
        additional_features = extract_additional_features(samples, sr)
        features = np.concatenate([mfcc_features, additional_features])
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        is_chirp = bool(prediction == 1)
        confidence = float(probability[1] if is_chirp else probability[0])
        
        return is_chirp, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)


def classify_event_is_chirp(
    event_chunks: List[bytes], 
    classifier_info: Optional[Dict[str, Any]], 
    duration_sec: float,
    config: Dict[str, Any],
    use_ml: bool = False,
    ml_model_info: Optional[Tuple] = None
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Classify if event is a chirp using either ML model or fingerprint method.
    
    This is the core classification function. It can use either:
    - ML model: Trained classifier (Random Forest or SVM)
    - Fingerprint: Multi-stage filtering with spectral similarity
    
    Args:
        event_chunks: List of raw PCM byte chunks (int16 little-endian)
        classifier_info: Dictionary with fingerprint info (for fingerprint method) or None
        duration_sec: Total event duration in seconds
        config: Configuration dictionary
        use_ml: If True, use ML model instead of fingerprint
        ml_model_info: ML model info tuple (if use_ml=True)
        
    Returns:
        Tuple of:
            - is_chirp (bool): True if classified as chirp
            - similarity (float or None): Similarity/confidence score (0-1)
            - confidence (float or None): Confidence score (0-1)
            - rejection_reason (str or None): Why it was rejected (if not chirp)
    """
    # Use ML model if requested and available
    if use_ml and ml_model_info is not None:
        return classify_event_is_chirp_ml(event_chunks, ml_model_info, duration_sec, config)
    
    # Fall back to fingerprint method
    fingerprint_info = classifier_info
    if fingerprint_info is None:
        return False, None, None, "no_classifier"
    
    chirp_cfg = config["chirp_classification"]
    freq_cfg = chirp_cfg["frequency_filtering"]
    temp_cfg = chirp_cfg["temporal_filtering"]
    conf_cfg = chirp_cfg.get("confidence", {})
    
    fp = fingerprint_info["fingerprint"]
    fft_size = fingerprint_info["fft_size"]
    sr = fingerprint_info["sample_rate"]
    
    # Find the best segment within the event
    best_chunks, best_similarity, segment_reason = find_best_chirp_segment(
        event_chunks, fingerprint_info, config
    )
    
    if best_chunks is None:
        return False, None, None, f"no_valid_segment_{segment_reason}"
    
    # Calculate duration of the best segment
    audio_cfg = config["audio"]
    chunk_duration = audio_cfg["chunk_duration"]
    best_segment_duration = len(best_chunks) * chunk_duration
    
    # Temporal filtering: reject if too long
    if best_segment_duration > temp_cfg["max_duration_sec"]:
        return False, best_similarity, None, f"duration_too_long_{best_segment_duration:.1f}s"
    
    # Use the best segment for classification
    event_spec = compute_event_spectrum_from_chunks(best_chunks, sr, fft_size)
    if event_spec is None:
        return False, None, None, "spectrum_computation_failed"
    
    # Frequency-domain filtering
    freq_resolution = sr / fft_size
    fan_noise_max_bin = int(freq_cfg["fan_noise_max_freq_hz"] / freq_resolution)
    chirp_min_bin = int(freq_cfg["chirp_min_freq_hz"] / freq_resolution)
    
    # Calculate energy in different frequency ranges
    total_energy = np.sum(event_spec)
    if total_energy > 0:
        low_freq_energy = np.sum(event_spec[:fan_noise_max_bin])
        high_freq_energy = np.sum(event_spec[chirp_min_bin:])
        low_freq_ratio = low_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
        
        # Reject if too much low-frequency energy
        low_freq_threshold = freq_cfg["low_freq_energy_threshold"]
        similarity_threshold = chirp_cfg["similarity_threshold"]
        if best_similarity >= similarity_threshold:
            low_freq_threshold = min(0.32, low_freq_threshold + 0.02)
        
        if low_freq_ratio > low_freq_threshold:
            return False, best_similarity, None, f"too_much_low_freq_{low_freq_ratio:.2f}"
        
        # Reject if insufficient high-frequency energy
        min_high_freq = freq_cfg.get("high_freq_energy_min_ratio", 0.1)
        if high_freq_ratio < min_high_freq:
            return False, best_similarity, None, f"insufficient_high_freq_{high_freq_ratio:.2f}"
    
    # Temporal envelope analysis
    chunk_rms_values = []
    for chunk in best_chunks:
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            chunk_rms_values.append(rms)
    
    energy_concentration_score = 0.5
    attack_decay_ratio = None
    spectral_centroid = None
    
    if len(chunk_rms_values) > 1:
        # Calculate energy concentration
        mid_point = len(chunk_rms_values) // 2
        first_half_energy = sum(r**2 for r in chunk_rms_values[:mid_point])
        second_half_energy = sum(r**2 for r in chunk_rms_values[mid_point:])
        total_chunk_energy = first_half_energy + second_half_energy
        
        if total_chunk_energy > 0:
            energy_concentration = first_half_energy / total_chunk_energy
            energy_concentration_score = energy_concentration
            
            energy_threshold = temp_cfg["energy_concentration_threshold"]
            if len(chunk_rms_values) <= 3:
                energy_threshold = max(0.3, energy_threshold - 0.2)
            
            if energy_concentration < energy_threshold:
                return False, best_similarity, None, f"energy_too_spread_{energy_concentration:.2f}"
        
        attack_decay_ratio = compute_attack_decay_ratio(chunk_rms_values)
    
    # Calculate spectral centroid
    spectral_centroid = compute_spectral_centroid(event_spec, sr, fft_size)
    
    # Spectral similarity
    sim = float(np.dot(fp, event_spec))
    
    # Check minimum similarity threshold
    similarity_threshold = chirp_cfg["similarity_threshold"]
    if sim < similarity_threshold:
        return False, sim, None, f"similarity_too_low_{sim:.3f}"
    
    # Calculate confidence score if enabled
    confidence = None
    if conf_cfg.get("enabled", True):
        sim_score = sim
        freq_score = min(1.0, spectral_centroid / 4000.0) if spectral_centroid else 0.5
        temp_score = energy_concentration_score
        
        weights = [
            conf_cfg.get("similarity_weight", 0.6),
            conf_cfg.get("frequency_weight", 0.2),
            conf_cfg.get("temporal_weight", 0.2)
        ]
        confidence = (
            weights[0] * sim_score +
            weights[1] * freq_score +
            weights[2] * temp_score
        )
    
    is_chirp = True
    return is_chirp, sim, confidence, None


# ============================================================================
# Classifier interface and implementations
# ============================================================================

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
        self._fingerprint_info = load_chirp_fingerprint(self.config)
    
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
        
        return classify_event_is_chirp(
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
        self._model_info = load_chirp_ml_model(self.config)
    
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
        
        return classify_event_is_chirp_ml(
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
