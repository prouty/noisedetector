#!/usr/bin/env python3
"""Audio analysis and calibration tools for the noise detector."""
import numpy as np
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import config_loader
import monitor


def analyze_audio_quality(clip_path: Path, config_path: Optional[Path] = None) -> Dict:
    """
    Analyze audio quality metrics: SNR, dynamic range, frequency response, DC offset.
    Returns a detailed report dictionary.
    """
    config = config_loader.load_config(config_path)
    
    if not clip_path.exists():
        return {"error": f"File not found: {clip_path}"}
    
    with wave.open(str(clip_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    samples = np.frombuffer(audio_data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    
    # DC offset check
    dc_offset = float(np.mean(samples))
    dc_offset_db = monitor.dbfs(abs(dc_offset))
    
    # Remove DC for analysis
    samples_dc_removed = samples - dc_offset
    
    # Dynamic range
    peak = float(np.max(np.abs(samples_dc_removed)))
    rms = float(np.sqrt(np.mean(samples_dc_removed ** 2)))
    peak_db = monitor.dbfs(peak)
    rms_db = monitor.dbfs(rms)
    dynamic_range_db = peak_db - rms_db
    
    # Estimate noise floor (using quietest 10% of samples)
    abs_samples = np.abs(samples_dc_removed)
    noise_floor = float(np.percentile(abs_samples, 10))
    noise_floor_db = monitor.dbfs(noise_floor)
    snr_estimate = peak_db - noise_floor_db if noise_floor_db > -np.inf else None
    
    # Frequency analysis
    fft_size = 2048
    if len(samples_dc_removed) >= fft_size:
        window = np.hanning(fft_size)
        chunk = samples_dc_removed[:fft_size] * window
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        
        # Find dominant frequencies
        peak_bin = np.argmax(spectrum[1:]) + 1  # Skip DC
        dominant_freq = float(freqs[peak_bin])
        
        # Energy distribution
        nyquist = sample_rate / 2
        low_band = freqs < 500
        mid_band = (freqs >= 500) & (freqs < 2000)
        high_band = freqs >= 2000
        
        total_energy = np.sum(spectrum[1:])  # Skip DC
        low_energy = np.sum(spectrum[1:][low_band[1:]])
        mid_energy = np.sum(spectrum[1:][mid_band[1:]])
        high_energy = np.sum(spectrum[1:][high_band[1:]])
        
        if total_energy > 0:
            low_ratio = float(low_energy / total_energy)
            mid_ratio = float(mid_energy / total_energy)
            high_ratio = float(high_energy / total_energy)
        else:
            low_ratio = mid_ratio = high_ratio = 0.0
    else:
        dominant_freq = None
        low_ratio = mid_ratio = high_ratio = None
    
    return {
        "file": str(clip_path),
        "sample_rate": sample_rate,
        "duration_sec": len(samples) / sample_rate,
        "dc_offset": dc_offset,
        "dc_offset_db": dc_offset_db,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "dynamic_range_db": dynamic_range_db,
        "noise_floor_db": noise_floor_db,
        "snr_estimate_db": snr_estimate,
        "dominant_frequency_hz": dominant_freq,
        "energy_distribution": {
            "low_band_ratio": low_ratio,
            "mid_band_ratio": mid_ratio,
            "high_band_ratio": high_ratio,
        }
    }


def check_dc_offset(clip_path: Path) -> Tuple[float, bool]:
    """
    Check for DC offset in audio file.
    Returns (dc_offset_db, is_problematic)
    Problematic if > -40 dBFS (significant DC component).
    """
    with wave.open(str(clip_path), "rb") as wf:
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    samples = np.frombuffer(audio_data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
    dc_offset = float(np.mean(samples))
    dc_offset_db = monitor.dbfs(abs(dc_offset))
    
    # DC offset is problematic if it's more than -40 dBFS
    is_problematic = dc_offset_db > -40.0
    
    return dc_offset_db, is_problematic


def validate_capture_levels(config_path: Optional[Path] = None, duration_sec: float = 5.0) -> Dict:
    """
    Validate that audio capture levels are appropriate.
    Checks for clipping, too-quiet signals, DC offset, and noise floor.
    """
    config = config_loader.load_config(config_path)
    
    print(f"Capturing {duration_sec} seconds for level validation...")
    print(f"Using device: {config['audio']['device']}")
    
    proc = monitor.start_arecord(config)
    if proc.stdout is None:
        return {"error": "Failed to open arecord stdout"}
    
    # Check if process started successfully
    import time
    time.sleep(0.1)  # Give arecord a moment to initialize
    if proc.poll() is not None:
        # Process exited immediately - check stderr
        stderr_output = ""
        if proc.stderr:
            stderr_output = proc.stderr.read().decode(errors="ignore")
        
        # Provide helpful error message for common "device busy" issue
        if "Device or resource busy" in stderr_output or "audio open error" in stderr_output:
            return {
                "error": "Audio device is busy. The noise-monitor service is likely running and has exclusive access to the device. Stop it first with 'make stop' or 'sudo systemctl stop noise-monitor', then run this check again."
            }
        
        return {"error": f"arecord failed to start. stderr: {stderr_output.strip()}"}
    
    audio_cfg = config["audio"]
    chunk_samples = int(audio_cfg["sample_rate"] * audio_cfg["chunk_duration"])
    chunk_bytes = chunk_samples * monitor.BYTES_PER_SAMPLE * audio_cfg["channels"]
    
    start = time.time()
    
    all_samples = []
    peak_values = []
    rms_values = []
    dc_offsets = []
    chunks_read = 0
    
    try:
        while time.time() - start < duration_sec:
            # Check if process is still running
            if proc.poll() is not None:
                stderr_output = ""
                if proc.stderr:
                    stderr_output = proc.stderr.read().decode(errors="ignore")
                
                # Provide helpful error message for common "device busy" issue
                if "Device or resource busy" in stderr_output or "audio open error" in stderr_output:
                    return {
                        "error": "Audio device is busy. The noise-monitor service is likely running and has exclusive access to the device. Stop it first with 'make stop' or 'sudo systemctl stop noise-monitor', then run this check again."
                    }
                
                return {"error": f"arecord process exited early. stderr: {stderr_output.strip()}"}
            
            data = proc.stdout.read(chunk_bytes)
            if not data:
                # No data available yet, wait a bit
                time.sleep(0.01)
                continue
            
            if len(data) < chunk_bytes:
                # Partial chunk - wait for more or break if process ended
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
            if len(samples) == 0:
                continue
            
            # Check for clipping (samples at or near full scale)
            peak = float(np.max(np.abs(samples)))
            if peak > 0.95:
                print(f"  ⚠️  WARNING: Clipping detected! Peak: {monitor.dbfs(peak):.1f} dBFS")
            
            # DC offset
            dc = float(np.mean(samples))
            dc_offsets.append(dc)
            
            # RMS
            rms = float(np.sqrt(np.mean(samples ** 2)))
            rms_values.append(monitor.dbfs(rms))
            peak_values.append(monitor.dbfs(peak))
            
            all_samples.extend(samples)
            chunks_read += 1
            
            # Print progress
            elapsed = time.time() - start
            if chunks_read % 10 == 0:
                print(f"  Captured {elapsed:.1f}s / {duration_sec:.1f}s ({chunks_read} chunks)...")
    
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
        elif proc.poll() is not None and proc.stderr:
            # Process exited - check for errors
            stderr_output = proc.stderr.read().decode(errors="ignore")
            if stderr_output:
                print(f"  arecord stderr: {stderr_output.strip()}")
    
    if not all_samples:
        return {"error": f"No audio captured after {duration_sec}s. Check device: {config['audio']['device']}"}
    
    print(f"  Captured {chunks_read} chunks ({len(all_samples)} samples)")
    
    all_samples = np.array(all_samples)
    
    # Analysis
    avg_rms_db = float(np.mean(rms_values))
    avg_peak_db = float(np.mean(peak_values))
    max_peak_db = float(np.max(peak_values))
    
    dc_offset = float(np.mean(dc_offsets))
    dc_offset_db = monitor.dbfs(abs(dc_offset))
    
    # Noise floor estimate (quietest 10%)
    abs_samples = np.abs(all_samples)
    noise_floor = float(np.percentile(abs_samples, 10))
    noise_floor_db = monitor.dbfs(noise_floor)
    
    # Recommendations
    recommendations = []
    
    if max_peak_db > -1.0:
        recommendations.append("⚠️  CLIPPING: Signal is too hot. Reduce input gain or check hardware.")
    elif max_peak_db < -30.0:
        recommendations.append("⚠️  TOO QUIET: Signal is very quiet. Increase input gain or check microphone placement.")
    elif max_peak_db < -20.0:
        recommendations.append("ℹ️  Signal is on the quiet side. Consider increasing gain if events are being missed.")
    
    if dc_offset_db > -40.0:
        recommendations.append("⚠️  DC OFFSET: Significant DC component detected. Check hardware or add high-pass filter.")
    
    if noise_floor_db > -60.0:
        recommendations.append("⚠️  HIGH NOISE FLOOR: Background noise is high. Check for electrical interference or improve shielding.")
    
    if avg_rms_db < -50.0:
        recommendations.append("ℹ️  Very quiet baseline. This is fine for quiet environments, but ensure events are loud enough to trigger.")
    
    return {
        "avg_rms_db": avg_rms_db,
        "avg_peak_db": avg_peak_db,
        "max_peak_db": max_peak_db,
        "dc_offset_db": dc_offset_db,
        "noise_floor_db": noise_floor_db,
        "dynamic_range_db": max_peak_db - noise_floor_db,
        "recommendations": recommendations,
        "status": "OK" if not recommendations else "NEEDS_ATTENTION"
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio analysis and calibration tools")
    parser.add_argument("command", choices=["analyze", "check-dc", "validate-levels"], help="Command to run")
    parser.add_argument("--clip", type=Path, help="Path to clip file (for analyze/check-dc)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration for validate-levels (seconds)")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        if not args.clip:
            print("Error: --clip required for analyze command")
            exit(1)
        result = analyze_audio_quality(args.clip, args.config)
        print(json.dumps(result, indent=2))
    
    elif args.command == "check-dc":
        if not args.clip:
            print("Error: --clip required for check-dc command")
            exit(1)
        dc_db, is_problematic = check_dc_offset(args.clip)
        print(f"DC Offset: {dc_db:.2f} dBFS")
        if is_problematic:
            print("⚠️  PROBLEMATIC: Significant DC offset detected")
        else:
            print("✓ OK: DC offset is within acceptable range")
    
    elif args.command == "validate-levels":
        result = validate_capture_levels(args.config, args.duration)
        if "error" in result:
            print(f"Error: {result['error']}")
            exit(1)
        
        print("\n" + "=" * 60)
        print("AUDIO LEVEL VALIDATION")
        print("=" * 60)
        print(f"Average RMS: {result['avg_rms_db']:.1f} dBFS")
        print(f"Average Peak: {result['avg_peak_db']:.1f} dBFS")
        print(f"Maximum Peak: {result['max_peak_db']:.1f} dBFS")
        print(f"DC Offset: {result['dc_offset_db']:.1f} dBFS")
        print(f"Noise Floor: {result['noise_floor_db']:.1f} dBFS")
        print(f"Dynamic Range: {result['dynamic_range_db']:.1f} dB")
        print()
        
        if result['recommendations']:
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  {rec}")
        else:
            print("✓ All checks passed - audio levels look good!")
        print()

