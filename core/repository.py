"""
Repository pattern for data persistence.

Single Responsibility: Handle file I/O operations.
"""
import csv
import datetime
import wave
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque

from config_loader import load_config


class EventRepository:
    """
    Repository for event data persistence.
    
    Single Responsibility: Event CSV file operations.
    """
    
    def __init__(self, config: dict):
        """
        Initialize event repository.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.events_file = Path(config["event_detection"]["events_file"])
        self._ensure_header()
    
    def _ensure_header(self) -> None:
        """Ensure CSV file has header row."""
        if not self.events_file.exists():
            self.events_file.parent.mkdir(parents=True, exist_ok=True)
            with self.events_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "start_timestamp",
                    "end_timestamp",
                    "duration_sec",
                    "max_peak_db",
                    "max_rms_db",
                    "baseline_rms_db",
                    "segment_file",
                    "segment_offset_sec",
                    "clip_file",
                    "is_chirp",
                    "chirp_similarity",
                    "confidence",
                    "rejection_reason",
                    "reviewed",
                ])
    
    def save(self, event: Dict[str, Any]) -> None:
        """
        Save event to CSV file.
        
        Args:
            event: Event dictionary with all required fields
        """
        try:
            with self.events_file.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event["start_timestamp"],
                    event["end_timestamp"],
                    f"{event['duration_sec']:.2f}",
                    f"{event['max_peak_db']:.2f}",
                    f"{event['max_rms_db']:.2f}",
                    f"{event['baseline_rms_db']:.2f}" if event["baseline_rms_db"] is not None else "",
                    str(event["segment_file"]),
                    f"{event['segment_offset_sec']:.2f}",
                    str(event.get("clip_file", "")),
                    "TRUE" if event.get("is_chirp") else "FALSE",
                    f"{event.get('chirp_similarity', 0.0):.3f}" if event.get("chirp_similarity") is not None else "",
                    f"{event.get('confidence', 0.0):.3f}" if event.get("confidence") is not None else "",
                    event.get("rejection_reason", ""),
                    event.get("reviewed", ""),
                ])
            
            # Print summary
            chirp_status = "CHIRP" if event.get("is_chirp") else "noise"
            conf_str = f", confidence={event.get('confidence', 0.0):.2f}" if event.get("confidence") is not None else ""
            print(
                f"[EVENT] Logged {chirp_status}: {event['start_timestamp']} – {event['end_timestamp']} "
                f"({event['duration_sec']:.2f}s, max_rms {event['max_rms_db']:.1f} dBFS{conf_str})"
            )
        except Exception as e:
            print(f"[ERROR] Failed to write events file {self.events_file}: {e}")


class SegmentRepository:
    """
    Repository for audio segment file operations.
    
    Single Responsibility: WAV segment file I/O.
    """
    
    INT16_FULL_SCALE = 32768.0
    BYTES_PER_SAMPLE = 2
    
    def __init__(self, config: dict):
        """
        Initialize segment repository.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.recording_config = config["recording"]
        self.audio_config = config["audio"]
        self.output_dir = Path(self.recording_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = self.audio_config["sample_rate"]
        self.channels = self.audio_config["channels"]
        self.segment_duration_sec = self.recording_config["segment_duration_sec"]
        self.segment_samples = int(self.sample_rate * self.segment_duration_sec)
        
        self._current_file: Optional[wave.Wave_write] = None
        self._current_path: Optional[Path] = None
        self._samples_written = 0
    
    def start_segment(self, start_time: datetime.datetime) -> None:
        """
        Start a new segment file.
        
        Args:
            start_time: Timestamp for segment start
        """
        if self._current_file is not None:
            self.close_segment()
        
        fname = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        fpath = self.output_dir / fname
        
        try:
            self._current_file = wave.open(str(fpath), "wb")
            self._current_file.setnchannels(self.channels)
            self._current_file.setsampwidth(self.BYTES_PER_SAMPLE)
            self._current_file.setframerate(self.sample_rate)
            self._current_path = fpath
            self._samples_written = 0
            print(f"[INFO] Started new segment: {fpath}")
        except Exception as e:
            raise OSError(f"Failed to open segment file {fpath}: {e}")
    
    def write_chunk(self, chunk_data: bytes, chunk_samples: int) -> None:
        """
        Write audio chunk to current segment.
        
        Args:
            chunk_data: Raw PCM bytes
            chunk_samples: Number of samples in chunk
        """
        if self._current_file is None:
            raise RuntimeError("No segment file open")
        
        self._current_file.writeframes(chunk_data)
        self._samples_written += chunk_samples
        
        # Check if we need to roll over
        if self._samples_written >= self.segment_samples:
            self.close_segment()
            import datetime
            self.start_segment(datetime.datetime.now())
    
    def close_segment(self) -> None:
        """Close current segment file."""
        if self._current_file is not None:
            try:
                self._current_file.close()
                if self._current_path:
                    print(f"[INFO] Closed segment: {self._current_path}")
            except Exception:
                pass
            finally:
                self._current_file = None
                self._current_path = None
                self._samples_written = 0
    
    def get_current_segment_path(self) -> Optional[Path]:
        """Get path of current segment file."""
        return self._current_path
    
    def get_current_segment_offset(self) -> float:
        """Get offset in seconds within current segment."""
        return self._samples_written / float(self.sample_rate)
    
    def save_clip(
        self,
        event_start_time: datetime.datetime,
        event_chunks: List[bytes]
    ) -> Path:
        """
        Save event clip to file.
        
        Args:
            event_start_time: Event start timestamp
            event_chunks: List of raw PCM byte chunks
            
        Returns:
            Path to saved clip file
        """
        clips_dir = Path(self.config["event_clips"]["clips_dir"])
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        fname = event_start_time.strftime("clip_%Y-%m-%d_%H-%M-%S.wav")
        fpath = clips_dir / fname
        
        with wave.open(str(fpath), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.BYTES_PER_SAMPLE)
            wf.setframerate(self.sample_rate)
            for chunk in event_chunks:
                wf.writeframes(chunk)
        
        print(f"[CLIP] Saved event clip: {fpath}")
        return fpath
    
    def __enter__(self):
        """Context manager entry."""
        import datetime
        self.start_segment(datetime.datetime.now())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_segment()

