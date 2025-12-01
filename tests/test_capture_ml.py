"""
Tests for ML-based capture decision system.

This test suite validates that:
1. ML capture model can be loaded
2. ML capture makes predictions correctly
3. Fallback to threshold works when model unavailable
4. Integration with monitor.py works
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Import modules under test
from scripts.capture_ml import (
    load_capture_model,
    should_capture_chunk,
    extract_capture_features
)
import config_loader
import monitor


class TestCaptureMLModel:
    """Test ML capture model loading and basic functionality."""
    
    def test_model_files_exist(self, data_dir):
        """Verify that model files exist if ML capture is configured."""
        model_file = data_dir / "capture_ml_model.joblib"
        scaler_file = data_dir / "capture_ml_scaler.joblib"
        metadata_file = data_dir / "capture_ml_metadata.json"
        
        # Check if any model files exist
        has_model = model_file.exists() and scaler_file.exists() and metadata_file.exists()
        
        if has_model:
            # If model exists, verify it can be loaded
            config = config_loader.load_config()
            model_info = load_capture_model(config)
            assert model_info is not None, "Model files exist but failed to load"
            
            model, scaler, metadata = model_info
            assert model is not None
            assert scaler is not None
            assert metadata is not None
            assert "model_type" in metadata
            assert "n_features" in metadata
        else:
            pytest.skip("ML capture model not trained yet. Run 'make train-capture-ml' first.")
    
    def test_model_loads_with_config(self, config, data_dir):
        """Test that model loads correctly from config."""
        model_info = load_capture_model(config)
        
        if model_info is None:
            pytest.skip("ML capture model not available")
        
        model, scaler, metadata = model_info
        
        # Verify model structure
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(scaler, 'transform')
        
        # Verify metadata
        assert isinstance(metadata, dict)
        assert "model_type" in metadata
        assert "n_features" in metadata
    
    def test_feature_extraction(self, config):
        """Test that feature extraction works on sample audio."""
        sample_rate = config["audio"]["sample_rate"]
        chunk_duration = config["audio"]["chunk_duration"]
        
        # Generate dummy audio (0.5s chunk)
        num_samples = int(sample_rate * chunk_duration)
        samples = np.random.randn(num_samples).astype(np.float32) * 0.1
        
        # Extract features
        features = extract_capture_features(samples, sample_rate)
        
        # Verify feature shape
        assert features.shape[0] > 0, "Features should not be empty"
        assert features.shape[0] >= 10, "Should have at least 10 features"
        assert np.all(np.isfinite(features)), "All features should be finite"
    
    def test_should_capture_with_model(self, config):
        """Test that should_capture_chunk works with a loaded model."""
        model_info = load_capture_model(config)
        
        if model_info is None:
            pytest.skip("ML capture model not available")
        
        sample_rate = config["audio"]["sample_rate"]
        
        # Test with random audio
        samples = np.random.randn(int(sample_rate * 0.5)).astype(np.float32) * 0.1
        should_capture, confidence = should_capture_chunk(
            samples, sample_rate, model_info, baseline_rms_db=-70.0
        )
        
        # Verify return types and values
        # Note: numpy may return np.bool_, so accept both
        assert isinstance(should_capture, (bool, np.bool_)), f"Expected bool, got {type(should_capture)}"
        assert isinstance(confidence, (float, np.floating)), f"Expected float, got {type(confidence)}"
        assert 0.0 <= float(confidence) <= 1.0, "Confidence should be between 0 and 1"
    
    def test_should_capture_fallback(self, config):
        """Test that fallback to threshold works when model is None."""
        sample_rate = config["audio"]["sample_rate"]
        
        # Test with loud audio (should trigger threshold)
        loud_samples = np.ones(int(sample_rate * 0.5), dtype=np.float32) * 0.5
        should_capture, confidence = should_capture_chunk(
            loud_samples, sample_rate, model_info=None, baseline_rms_db=-70.0
        )
        
        # Should use threshold fallback
        # Note: numpy may return np.bool_, so convert for comparison
        assert isinstance(should_capture, (bool, np.bool_)), f"Expected bool, got {type(should_capture)}"
        assert isinstance(confidence, (float, np.floating)), f"Expected float, got {type(confidence)}"
        # Loud audio should trigger capture
        assert bool(should_capture) is True, "Loud audio should trigger threshold-based capture"
    
    def test_model_metadata_valid(self, config, data_dir):
        """Test that model metadata contains expected information."""
        model_info = load_capture_model(config)
        
        if model_info is None:
            pytest.skip("ML capture model not available")
        
        _, _, metadata = model_info
        
        # Check required metadata fields
        required_fields = ["model_type", "n_features", "n_samples", "n_positive", "n_negative"]
        for field in required_fields:
            assert field in metadata, f"Metadata missing required field: {field}"
        
        # Verify metadata values are reasonable
        assert metadata["n_features"] > 0
        assert metadata["n_samples"] > 0
        assert metadata["n_positive"] >= 0
        assert metadata["n_negative"] >= 0
        assert metadata["n_positive"] + metadata["n_negative"] == metadata["n_samples"]


class TestCaptureMLIntegration:
    """Test integration with monitor.py."""
    
    def test_monitor_imports_capture_ml(self):
        """Test that monitor.py can import capture_ml module."""
        assert hasattr(monitor, 'CAPTURE_ML_AVAILABLE'), "monitor should have CAPTURE_ML_AVAILABLE flag"
        assert hasattr(monitor, 'load_capture_model'), "monitor should import load_capture_model"
        assert hasattr(monitor, 'should_capture_chunk'), "monitor should import should_capture_chunk"
    
    def test_config_has_ml_capture_option(self, config):
        """Test that config has ML capture options."""
        event_detection = config.get("event_detection", {})
        assert "use_ml_capture" in event_detection, "Config should have use_ml_capture option"
        assert "capture_ml_model_file" in event_detection, "Config should have capture_ml_model_file"
        assert "capture_ml_scaler_file" in event_detection, "Config should have capture_ml_scaler_file"
        assert "capture_ml_metadata_file" in event_detection, "Config should have capture_ml_metadata_file"
    
    def test_model_paths_in_config(self, config):
        """Test that model file paths in config are valid."""
        event_detection = config.get("event_detection", {})
        model_file = Path(event_detection.get("capture_ml_model_file", "data/capture_ml_model.joblib"))
        scaler_file = Path(event_detection.get("capture_ml_scaler_file", "data/capture_ml_scaler.joblib"))
        metadata_file = Path(event_detection.get("capture_ml_metadata_file", "data/capture_ml_metadata.json"))
        
        # If model exists, all files should exist
        if model_file.exists():
            assert scaler_file.exists(), f"Scaler file should exist: {scaler_file}"
            assert metadata_file.exists(), f"Metadata file should exist: {metadata_file}"


class TestCaptureMLValidation:
    """Test validation that ML capture is properly configured."""
    
    def test_validate_ml_capture_setup(self, config, data_dir):
        """
        Comprehensive test to validate ML capture is properly set up.
        
        This is the main validation test that checks:
        1. Config has use_ml_capture option
        2. Model files exist
        3. Model can be loaded
        4. Model makes predictions
        """
        event_detection = config.get("event_detection", {})
        use_ml_capture = event_detection.get("use_ml_capture", False)
        
        if not use_ml_capture:
            pytest.skip("ML capture not enabled in config (use_ml_capture: false)")
        
        # Check model files exist
        model_file = data_dir / "capture_ml_model.joblib"
        scaler_file = data_dir / "capture_ml_scaler.joblib"
        metadata_file = data_dir / "capture_ml_metadata.json"
        
        assert model_file.exists(), f"Model file not found: {model_file}"
        assert scaler_file.exists(), f"Scaler file not found: {scaler_file}"
        assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
        
        # Load model
        model_info = load_capture_model(config)
        assert model_info is not None, "Failed to load ML capture model"
        
        model, scaler, metadata = model_info
        
        # Verify model can make predictions
        sample_rate = config["audio"]["sample_rate"]
        test_samples = np.random.randn(int(sample_rate * 0.5)).astype(np.float32) * 0.1
        
        should_capture, confidence = should_capture_chunk(
            test_samples, sample_rate, model_info, baseline_rms_db=-70.0
        )
        
        assert isinstance(should_capture, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Print validation summary
        print(f"\n✓ ML Capture Validation Summary:")
        print(f"  Model type: {metadata.get('model_type', 'unknown')}")
        print(f"  Features: {metadata.get('n_features', 'unknown')}")
        print(f"  Training samples: {metadata.get('n_samples', 'unknown')}")
        print(f"  Training accuracy: {metadata.get('cv_accuracy', 0):.3f}")
        print(f"  Test prediction: should_capture={should_capture}, confidence={confidence:.3f}")

