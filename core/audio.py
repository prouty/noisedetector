"""
Audio capture abstraction.

Single Responsibility: Handle audio input from hardware.
"""
import subprocess
from dataclasses import dataclass
from typing import Optional, Iterator
import numpy as np

from config_loader import load_config


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    samples: np.ndarray  # Normalized float32 samples [-1.0, 1.0]
    raw_bytes: bytes  # Original raw PCM bytes (int16)
    sample_rate: int
    timestamp: float  # Unix timestamp
    
    @property
    def duration_sec(self) -> float:
        """Duration of chunk in seconds."""
        return len(self.samples) / self.sample_rate
    
    @property
    def peak(self) -> float:
        """Peak amplitude."""
        return float(np.max(np.abs(self.samples)))
    
    @property
    def rms(self) -> float:
        """RMS amplitude."""
        return float(np.sqrt(np.mean(self.samples ** 2)))


class AudioCapture:
    """
    Handles audio capture from ALSA arecord.
    
    Single Responsibility: Audio I/O operations.
    """
    
    INT16_FULL_SCALE = 32768.0
    BYTES_PER_SAMPLE = 2
    
    def __init__(self, config: dict):
        """
        Initialize audio capture.
        
        Args:
            config: Configuration dictionary with audio settings
        """
        self.config = config
        self.audio_config = config["audio"]
        self.sample_rate = self.audio_config["sample_rate"]
        self.channels = self.audio_config["channels"]
        self.chunk_duration = self.audio_config["chunk_duration"]
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.chunk_bytes = self.chunk_samples * self.BYTES_PER_SAMPLE * self.channels
        
        self._process: Optional[subprocess.Popen] = None
        self._dc_offset_ema = 0.0
    
    def start(self) -> None:
        """Start audio capture process."""
        if self._process is not None:
            raise RuntimeError("Audio capture already started")
        
        device = self.audio_config["device"]
        if not device or not isinstance(device, str):
            raise ValueError(
                f"Invalid audio device configuration: {device}. "
                f"Expected string like 'plughw:CARD=Device,DEV=0'"
            )
        
        cmd = [
            "arecord",
            "-D", device,
            "-f", self.audio_config["sample_format"],
            "-r", str(self.sample_rate),
            "-c", str(self.channels),
            "-q",
            "-t", "raw"
        ]
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "arecord command not found. Install alsa-utils: "
                "sudo apt-get install alsa-utils"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to start arecord process. Command: {' '.join(cmd)}. Error: {e}"
            )
        
        # Give process a moment to initialize
        import time
        time.sleep(0.1)
        
        if self._process.poll() is not None:
            stderr_msg = ""
            if self._process.stderr:
                stderr_msg = self._process.stderr.read().decode(errors="ignore").strip()
            
            error_hints = {
                "Device or resource busy": "Audio device is in use. Stop noise-monitor service: 'make stop'",
                "No such file or directory": f"Audio device '{device}' not found. Check with 'arecord -l'",
                "Permission denied": "No permission to access audio device. Add user to audio group: 'sudo usermod -a -G audio $USER'",
                "Invalid argument": f"Invalid audio device or format. Device: {device}, Format: {self.audio_config['sample_format']}"
            }
            
            hint = ""
            for key, msg in error_hints.items():
                if key in stderr_msg:
                    hint = f" Hint: {msg}"
                    break
            
            raise RuntimeError(
                f"arecord failed to start. Device: {device}. Error: {stderr_msg}.{hint}"
            )
    
    def read_chunk(self) -> Optional[AudioChunk]:
        """
        Read next audio chunk.
        
        Returns:
            AudioChunk or None if stream ended
        """
        if self._process is None:
            raise RuntimeError("Audio capture not started")
        
        if self._process.stdout is None:
            return None
        
        data = self._process.stdout.read(self.chunk_bytes)
        
        if not data or len(data) < self.chunk_bytes:
            return None
        
        # Convert raw bytes to numpy array
        samples = np.frombuffer(data, dtype="<i2")  # little-endian int16
        
        if samples.size == 0:
            return None
        
        # Normalize to float32 [-1.0, 1.0]
        float_samples = samples.astype(np.float32) / self.INT16_FULL_SCALE
        
        # Remove DC offset (exponential moving average)
        alpha = 0.001
        self._dc_offset_ema = (
            alpha * float(np.mean(float_samples)) + 
            (1 - alpha) * self._dc_offset_ema
        )
        float_samples = float_samples - self._dc_offset_ema
        
        import time
        return AudioChunk(
            samples=float_samples,
            raw_bytes=data,  # Store original raw bytes
            sample_rate=self.sample_rate,
            timestamp=time.time()
        )
    
    def is_running(self) -> bool:
        """Check if capture process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def stop(self) -> None:
        """Stop audio capture process."""
        if self._process is None:
            return
        
        if self._process.poll() is None:
            self._process.terminate()
            import time
            time.sleep(0.1)
            if self._process.poll() is None:
                self._process.kill()
        
        self._process = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

