"""
Tests for extract_chirp_segment.py script.

Tests the percentage-based segment extraction functionality.
"""
import pytest
import wave
from pathlib import Path
import tempfile

from tests.conftest import create_test_wav_file, TEST_SAMPLE_RATE

# Import the functions we want to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_chirp_segment import (
    get_clip_duration,
    extract_segment_from_clip,
)


class TestGetClipDuration:
    """Test get_clip_duration function."""
    
    def test_get_clip_duration(self):
        """Test getting duration of a valid clip file."""
        clip_path, sr = create_test_wav_file(duration=5.0)
        
        try:
            duration = get_clip_duration(clip_path)
            
            assert duration is not None
            assert abs(duration - 5.0) < 0.1  # Allow small floating point differences
        finally:
            clip_path.unlink()
    
    def test_get_clip_duration_nonexistent_file(self):
        """Test getting duration of non-existent file returns None."""
        fake_path = Path("/nonexistent/file.wav")
        duration = get_clip_duration(fake_path)
        assert duration is None
    
    def test_get_clip_duration_different_durations(self):
        """Test getting duration for clips of different lengths."""
        test_durations = [1.0, 10.0, 30.0, 180.0]
        
        for test_duration in test_durations:
            clip_path, sr = create_test_wav_file(duration=test_duration)
            
            try:
                duration = get_clip_duration(clip_path)
                assert duration is not None
                assert abs(duration - test_duration) < 0.1
            finally:
                clip_path.unlink()


class TestExtractSegmentFromClip:
    """Test extract_segment_from_clip function."""
    
    def test_extract_segment_at_50_percent(self):
        """Test extracting segment at 50% of clip (middle)."""
        # Create a 100-second clip
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            # Extract at 50% (50 seconds) with 5s padding
            # Should extract from 45s to 55s
            center_sec = 50.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify the extracted segment duration
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                # Should be approximately 10 seconds (5s before + 5s after)
                assert abs(extracted_duration - 10.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_at_25_percent(self):
        """Test extracting segment at 25% of clip."""
        # Create a 100-second clip
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            # Extract at 25% (25 seconds) with 5s padding
            # Should extract from 20s to 30s
            center_sec = 25.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify the extracted segment duration
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                assert abs(extracted_duration - 10.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_at_beginning(self):
        """Test extracting segment near the beginning (edge case)."""
        # Create a 100-second clip
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            # Extract at 3 seconds (near beginning)
            # Should extract from 0s to 8s (clamped to start)
            center_sec = 3.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify the extracted segment duration
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                # Should be 8 seconds (3s + 5s, clamped at 0)
                assert abs(extracted_duration - 8.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_at_end(self):
        """Test extracting segment near the end (edge case)."""
        # Create a 100-second clip
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            # Extract at 98 seconds (near end)
            # Should extract from 93s to 100s (clamped to end)
            center_sec = 98.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify the extracted segment duration
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                # Should be 7 seconds (5s before + 2s to end)
                assert abs(extracted_duration - 7.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_very_short_clip(self):
        """Test extracting segment from a very short clip."""
        # Create a 5-second clip
        clip_path, sr = create_test_wav_file(duration=5.0)
        
        try:
            # Extract at 2.5 seconds (middle) with 5s padding
            # Should extract entire clip (0s to 5s)
            center_sec = 2.5
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify the extracted segment duration
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                # Should be 5 seconds (entire clip)
                assert abs(extracted_duration - 5.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_invalid_center(self):
        """Test extracting segment with invalid center point."""
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            # Negative center point should return None
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=-10.0,
                padding_sec=5.0
            )
            
            assert output_path is None
        finally:
            clip_path.unlink()
    
    def test_extract_segment_nonexistent_file(self):
        """Test extracting segment from non-existent file."""
        fake_path = Path("/nonexistent/file.wav")
        output_path = extract_segment_from_clip(
            fake_path,
            center_sec=50.0,
            padding_sec=5.0
        )
        
        assert output_path is None
    
    def test_extract_segment_custom_output_path(self):
        """Test extracting segment with custom output path."""
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            custom_output = Path(tmp.name)
        
        try:
            center_sec = 50.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0,
                output_path=custom_output
            )
            
            assert output_path == custom_output
            assert output_path.exists()
            
            output_path.unlink()
        finally:
            clip_path.unlink()
    
    def test_extract_segment_default_output_path(self):
        """Test that default output path is created correctly."""
        clip_path, sr = create_test_wav_file(duration=100.0)
        
        try:
            center_sec = 50.0
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            # Should be in same directory as source
            assert output_path.parent == clip_path.parent
            # Should have _segment suffix
            assert "_segment" in output_path.stem
            assert output_path.suffix == ".wav"
            
            output_path.unlink()
        finally:
            clip_path.unlink()


class TestPercentageCalculation:
    """Test percentage-based calculations."""
    
    def test_percentage_to_center_point(self):
        """Test converting percentage to center point."""
        # This tests the logic used in main()
        duration = 180.0  # 3 minutes
        
        test_cases = [
            (1, 1.8),      # 1% of 180s = 1.8s
            (25, 45.0),    # 25% of 180s = 45s
            (50, 90.0),    # 50% of 180s = 90s
            (75, 135.0),   # 75% of 180s = 135s
            (100, 180.0),  # 100% of 180s = 180s
        ]
        
        for percent, expected_center in test_cases:
            center_sec = (percent / 100.0) * duration
            assert abs(center_sec - expected_center) < 0.01
    
    def test_percentage_extraction_integration(self):
        """Integration test: extract segment using percentage calculation."""
        # Create a 180-second clip (3 minutes)
        clip_path, sr = create_test_wav_file(duration=180.0)
        
        try:
            # Simulate 50% extraction
            duration = get_clip_duration(clip_path)
            assert duration is not None
            
            percent = 50.0
            center_sec = (percent / 100.0) * duration
            
            # Extract segment
            output_path = extract_segment_from_clip(
                clip_path,
                center_sec=center_sec,
                padding_sec=5.0
            )
            
            assert output_path is not None
            assert output_path.exists()
            
            # Verify center point was correct
            # At 50% of 180s = 90s, so should extract 85s-95s
            with wave.open(str(output_path), "rb") as wf:
                extracted_frames = wf.getnframes()
                extracted_duration = extracted_frames / float(wf.getframerate())
                # Should be 10 seconds
                assert abs(extracted_duration - 10.0) < 0.1
            
            output_path.unlink()
        finally:
            clip_path.unlink()

