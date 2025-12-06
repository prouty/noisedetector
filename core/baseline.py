"""
Baseline tracking for noise level estimation.

Single Responsibility: Manage baseline noise level calculation and tracking.
"""
from typing import Optional, List
import numpy as np
import json
from pathlib import Path

import config_loader
import baseline as baseline_module


class BaselineTracker:
    """
    Tracks and updates baseline noise level.
    
    Single Responsibility: Baseline calculation and management.
    """
    
    def __init__(self, config: dict):
        """
        Initialize baseline tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.event_detection_config = config["event_detection"]
        self.baseline_window_chunks = self.event_detection_config["baseline_window_chunks"]
        
        # State
        self._baseline_rms_db: Optional[float] = None
        self._baseline_window: List[float] = []
        self._in_event = False
        
        # Load initial baseline
        self._load_initial_baseline()
    
    def _load_initial_baseline(self) -> None:
        """Load initial baseline from file."""
        baseline_file = Path(self.event_detection_config["baseline_file"])
        
        if not baseline_file.exists():
            return
        
        try:
            with baseline_file.open() as f:
                data = json.load(f)
            
            # Handle both single baseline and history array formats
            if isinstance(data, list):
                if len(data) == 0:
                    return
                latest = data[-1]
            else:
                latest = data
            
            rms_db = float(latest.get("rms_db", 0))
            
            # Validate reasonable range
            if -100 < rms_db < 0:
                self._baseline_rms_db = rms_db
                print(f"[INFO] Loaded baseline RMS: {rms_db:.1f} dBFS")
        except Exception:
            pass  # Use rolling baseline only
    
    def update(self, rms_db: float, in_event: bool) -> None:
        """
        Update baseline with new RMS value.
        
        Args:
            rms_db: RMS level in dBFS
            in_event: Whether currently in an event (baseline not updated during events)
        """
        self._in_event = in_event
        
        # Only update baseline window when not in event
        if not in_event:
            self._baseline_window.append(rms_db)
            if len(self._baseline_window) > self.baseline_window_chunks:
                self._baseline_window.pop(0)
            
            # Calculate baseline as 20th percentile
            if self._baseline_window:
                valid_baseline = [
                    v for v in self._baseline_window 
                    if np.isfinite(v) and v > -100
                ]
                if valid_baseline:
                    self._baseline_rms_db = float(np.percentile(valid_baseline, 20))
    
    @property
    def baseline_rms_db(self) -> Optional[float]:
        """Get current baseline RMS in dBFS."""
        return self._baseline_rms_db
    
    def get_threshold_db(self) -> float:
        """
        Get detection threshold (baseline + offset).
        
        Returns:
            Threshold in dBFS
        """
        threshold_offset = self.event_detection_config["threshold_above_baseline_db"]
        effective_baseline = self._baseline_rms_db if self._baseline_rms_db is not None else -50.0
        return effective_baseline + threshold_offset
    
    def reset_event_state(self) -> None:
        """Reset event state (called when event ends)."""
        self._in_event = False

