"""
Event detection logic.

Single Responsibility: Detect noise events above threshold.
"""
import datetime
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass

from .audio import AudioChunk
from .baseline import BaselineTracker


@dataclass
class Event:
    """Represents a detected noise event."""
    start_time: datetime.datetime
    end_time: datetime.datetime
    max_peak_db: float
    max_rms_db: float
    baseline_rms_db: Optional[float]
    chunks: List[bytes]  # Raw PCM chunks
    actual_start_idx: int  # Index where real event starts (after pre-roll)


class EventDetector:
    """
    Detects noise events above baseline threshold.
    
    Single Responsibility: Event detection state machine.
    """
    
    def __init__(self, config: dict, baseline_tracker: BaselineTracker):
        """
        Initialize event detector.
        
        Args:
            config: Configuration dictionary
            baseline_tracker: BaselineTracker instance
        """
        self.config = config
        self.baseline_tracker = baseline_tracker
        self.event_detection_config = config["event_detection"]
        self.event_clips_config = config["event_clips"]
        
        chunk_duration = config["audio"]["chunk_duration"]
        pre_roll_sec = self.event_clips_config["pre_roll_sec"]
        self.pre_roll_chunks = int(pre_roll_sec / chunk_duration)
        
        # State
        self._in_event = False
        self._event_start_time: Optional[datetime.datetime] = None
        self._event_end_time: Optional[datetime.datetime] = None
        self._event_max_peak_db: Optional[float] = None
        self._event_max_rms_db: Optional[float] = None
        self._event_baseline_at_start: Optional[float] = None
        self._pre_roll_buffer = deque(maxlen=self.pre_roll_chunks)
        self._event_chunks: Optional[List[bytes]] = None
        self._event_actual_start_idx: Optional[int] = None
    
    def process_chunk(self, chunk: AudioChunk, chunk_data: bytes) -> Optional[Event]:
        """
        Process audio chunk and detect events.
        
        Args:
            chunk: AudioChunk with processed samples
            chunk_data: Raw PCM bytes for saving
            
        Returns:
            Event if event just ended, None otherwise
        """
        import monitor
        
        # Calculate RMS in dBFS
        import monitor as monitor_module
        rms_db = monitor_module.dbfs(chunk.rms)
        peak_db = monitor_module.dbfs(chunk.peak)
        
        # Update baseline
        self.baseline_tracker.update(rms_db, self._in_event)
        
        # Get threshold
        threshold_db = self.baseline_tracker.get_threshold_db()
        
        # Maintain pre-roll buffer when not in event
        if not self._in_event:
            self._pre_roll_buffer.append(chunk_data)
        
        # Event state machine
        if not self._in_event:
            # Check if we should start an event
            if rms_db > threshold_db:
                self._start_event(chunk, rms_db, peak_db)
                return None
        else:
            # Already in event - update stats
            self._event_end_time = datetime.datetime.fromtimestamp(chunk.timestamp)
            if peak_db > self._event_max_peak_db:
                self._event_max_peak_db = peak_db
            if rms_db > self._event_max_rms_db:
                self._event_max_rms_db = rms_db
            
            # Check if event ended
            if rms_db <= threshold_db:
                # Event ended - check if it meets minimum duration
                duration_sec = (self._event_end_time - self._event_start_time).total_seconds()
                min_duration = self.event_detection_config["min_event_duration_sec"]
                
                if duration_sec >= min_duration:
                    event = self._create_event()
                    self._reset_state()
                    return event
                else:
                    # Too short - discard
                    self._reset_state()
                    return None
            else:
                # Still in event - append chunk
                if self._event_chunks is not None:
                    self._event_chunks.append(chunk_data)
        
        return None
    
    def _start_event(
        self,
        chunk: AudioChunk,
        rms_db: float,
        peak_db: float
    ) -> None:
        """Start a new event."""
        self._in_event = True
        self._event_start_time = datetime.datetime.fromtimestamp(chunk.timestamp)
        self._event_end_time = self._event_start_time
        self._event_max_peak_db = peak_db
        self._event_max_rms_db = rms_db
        self._event_baseline_at_start = self.baseline_tracker.baseline_rms_db
        
        # Build initial clip buffer: pre-roll + current chunk
        if self._pre_roll_buffer:
            self._event_chunks = list(self._pre_roll_buffer)
        else:
            self._event_chunks = []
        self._event_chunks.append(chunk_data)
        
        # Track where actual event starts (after pre-roll)
        self._event_actual_start_idx = len(self._event_chunks) - 1
    
    def _create_event(self) -> Event:
        """Create Event object from current state."""
        return Event(
            start_time=self._event_start_time,
            end_time=self._event_end_time,
            max_peak_db=self._event_max_peak_db or 0.0,
            max_rms_db=self._event_max_rms_db or 0.0,
            baseline_rms_db=self._event_baseline_at_start,
            chunks=self._event_chunks or [],
            actual_start_idx=self._event_actual_start_idx or 0
        )
    
    def _reset_state(self) -> None:
        """Reset event detection state."""
        self._in_event = False
        self._event_start_time = None
        self._event_end_time = None
        self._event_max_peak_db = None
        self._event_max_rms_db = None
        self._event_baseline_at_start = None
        self._event_chunks = None
        self._event_actual_start_idx = None
        self.baseline_tracker.reset_event_state()
    
    @property
    def in_event(self) -> bool:
        """Check if currently in an event."""
        return self._in_event

