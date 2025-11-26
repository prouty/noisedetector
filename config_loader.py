#!/usr/bin/env python3
"""Configuration loader for noise detector."""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def get_default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "audio": {
            "device": "plughw:CARD=Device,DEV=0",
            "sample_rate": 16000,
            "channels": 1,
            "sample_format": "S16_LE",
            "chunk_duration": 0.5,
            "dc_offset_removal": True,
            "high_pass_filter_hz": 20
        },
        "recording": {
            "output_dir": "clips",
            "segment_duration_sec": 300
        },
        "event_detection": {
            "baseline_file": "baseline.json",
            "events_file": "events.csv",
            "threshold_above_baseline_db": 10.0,
            "min_event_duration_sec": 0.5,
            "baseline_window_chunks": 120
        },
        "event_clips": {
            "clips_dir": "clips",
            "pre_roll_sec": 2.0
        },
        "chirp_classification": {
            "fingerprint_file": "chirp_fingerprint.json",
            "similarity_threshold": 0.8,
            "frequency_filtering": {
                "fan_noise_max_freq_hz": 500,
                "chirp_min_freq_hz": 1000,
                "low_freq_energy_threshold": 0.3,
                "high_freq_energy_min_ratio": 0.1
            },
            "temporal_filtering": {
                "max_duration_sec": 2.0,
                "energy_concentration_threshold": 0.5
            },
            "confidence": {
                "enabled": True,
                "similarity_weight": 0.6,
                "frequency_weight": 0.2,
                "temporal_weight": 0.2
            }
        }
    }


def validate_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate configuration structure and values."""
    defaults = get_default_config()
    
    # Check required top-level keys
    for key in defaults.keys():
        if key not in config:
            return False, f"Missing required config section: {key}"
    
    # Validate audio settings
    audio = config.get("audio", {})
    if not isinstance(audio.get("sample_rate"), int) or audio.get("sample_rate") <= 0:
        return False, "audio.sample_rate must be a positive integer"
    if audio.get("chunk_duration") <= 0:
        return False, "audio.chunk_duration must be positive"
    
    # Validate event detection
    event = config.get("event_detection", {})
    if event.get("threshold_above_baseline_db") < 0:
        return False, "event_detection.threshold_above_baseline_db must be non-negative"
    if event.get("min_event_duration_sec") <= 0:
        return False, "event_detection.min_event_duration_sec must be positive"
    
    # Validate chirp classification
    chirp = config.get("chirp_classification", {})
    if not 0 <= chirp.get("similarity_threshold", 0) <= 1:
        return False, "chirp_classification.similarity_threshold must be between 0 and 1"
    
    freq = chirp.get("frequency_filtering", {})
    if not 0 <= freq.get("low_freq_energy_threshold", 0) <= 1:
        return False, "chirp_classification.frequency_filtering.low_freq_energy_threshold must be between 0 and 1"
    if not 0 <= freq.get("high_freq_energy_min_ratio", 0) <= 1:
        return False, "chirp_classification.frequency_filtering.high_freq_energy_min_ratio must be between 0 and 1"
    
    temp = chirp.get("temporal_filtering", {})
    if temp.get("max_duration_sec") <= 0:
        return False, "chirp_classification.temporal_filtering.max_duration_sec must be positive"
    if not 0 <= temp.get("energy_concentration_threshold", 0) <= 1:
        return False, "chirp_classification.temporal_filtering.energy_concentration_threshold must be between 0 and 1"
    
    conf = chirp.get("confidence", {})
    if conf.get("enabled"):
        weights = [conf.get("similarity_weight", 0), conf.get("frequency_weight", 0), conf.get("temporal_weight", 0)]
        if abs(sum(weights) - 1.0) > 0.01:
            return False, "chirp_classification.confidence weights must sum to approximately 1.0"
    
    return True, None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file, merging with defaults.
    
    Args:
        config_path: Path to config file. If None, looks for config.json in current directory.
    
    Returns:
        Merged configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file specified but not found.
        ValueError: If config is invalid.
    """
    defaults = get_default_config()
    
    if config_path is None:
        config_path = Path("config.json")
    
    if not config_path.exists():
        print(f"[INFO] Config file {config_path} not found, using defaults")
        return defaults
    
    try:
        with config_path.open() as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    # Deep merge with defaults
    merged = _deep_merge(defaults, config)
    
    # Validate
    is_valid, error_msg = validate_config(merged)
    if not is_valid:
        raise ValueError(f"Invalid configuration: {error_msg}")
    
    print(f"[INFO] Loaded configuration from {config_path}")
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Example: get_config_value(config, "audio.sample_rate")
    """
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

