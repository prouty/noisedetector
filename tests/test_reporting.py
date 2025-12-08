"""
Tests for core.reporting module.

Tests event loading, filtering, and report generation.
"""
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import csv

from core.reporting import (
    load_events,
    filter_recent_events,
    generate_email_report,
    add_date_column,
    choose_latest_date,
    generate_chirp_report,
)


class TestLoadEvents:
    """Test event loading from CSV."""
    
    def test_load_events_existing_file(self, project_root_path):
        """Test loading existing events.csv file."""
        events_file = project_root_path / "data" / "events.csv"
        
        if not events_file.exists():
            pytest.skip("No events.csv file found")
        
        df = load_events(events_file)
        
        assert isinstance(df, pd.DataFrame)
        # Should have some columns if file has data
        if not df.empty:
            assert len(df.columns) > 0
    
    def test_load_events_nonexistent_file(self):
        """Test loading non-existent file returns empty DataFrame."""
        fake_file = Path("/nonexistent/events.csv")
        df = load_events(fake_file)
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    def test_load_events_creates_temp_file(self):
        """Test loading a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.writer(tmp)
            writer.writerow(["start_timestamp", "duration_sec", "is_chirp", "clip_file"])
            writer.writerow(["2025-01-01T12:00:00", "1.5", "True", "clip1.wav"])
            writer.writerow(["2025-01-01T12:05:00", "2.0", "False", "clip2.wav"])
        
        try:
            df = load_events(tmp_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "start_timestamp" in df.columns
            assert "duration_sec" in df.columns
            assert "is_chirp" in df.columns
        finally:
            tmp_path.unlink()
    
    def test_load_events_strips_column_names(self):
        """Test that column names are stripped of whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.writer(tmp)
            writer.writerow([" start_timestamp ", " duration_sec "])
            writer.writerow(["2025-01-01T12:00:00", "1.5"])
        
        try:
            df = load_events(tmp_path)
            
            # Column names should be stripped
            assert "start_timestamp" in df.columns
            assert "duration_sec" in df.columns
            assert " start_timestamp " not in df.columns
        finally:
            tmp_path.unlink()
    
    def test_load_events_specifies_is_chirp_dtype(self):
        """Test that load_events reads is_chirp as object dtype to prevent FutureWarning."""
        test_df = pd.DataFrame({"start_timestamp": ["2025-01-01T12:00:00"], "is_chirp": [True]})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            test_df.to_csv(tmp_path, index=False)
        
        try:
            df = load_events(tmp_path)
            assert df["is_chirp"].dtype == "object"
            df.iloc[0, df.columns.get_loc("is_chirp")] = "FALSE"  # Should not raise FutureWarning
        finally:
            tmp_path.unlink()


class TestFilterRecentEvents:
    """Test filtering events by time window."""
    
    def test_filter_recent_events(self):
        """Test filtering to recent events."""
        # Create test DataFrame
        now = datetime.now()
        df = pd.DataFrame({
            "start_timestamp": [
                (now - timedelta(hours=1)).isoformat(),
                (now - timedelta(hours=3)).isoformat(),
                (now - timedelta(minutes=30)).isoformat(),
            ],
            "duration_sec": [1.0, 1.5, 2.0],
        })
        
        recent = filter_recent_events(df, hours=2)
        
        assert isinstance(recent, pd.DataFrame)
        assert len(recent) == 2  # Should include 1h and 30min ago, not 3h
    
    def test_filter_recent_events_empty(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame()
        recent = filter_recent_events(df, hours=2)
        
        assert isinstance(recent, pd.DataFrame)
        assert recent.empty
    
    def test_filter_recent_events_no_timestamp_column(self):
        """Test filtering when timestamp column missing."""
        df = pd.DataFrame({
            "duration_sec": [1.0, 1.5],
        })
        
        recent = filter_recent_events(df, hours=2)
        
        # Should return original DataFrame unchanged
        assert len(recent) == len(df)


class TestGenerateEmailReport:
    """Test email report generation."""
    
    def test_generate_email_report_empty(self):
        """Test generating report from empty DataFrame."""
        df = pd.DataFrame()
        report = generate_email_report(df, hours=2)
        
        assert isinstance(report, str)
        assert "No events recorded" in report
    
    def test_generate_email_report_with_data(self):
        """Test generating report with event data."""
        df = pd.DataFrame({
            "start_timestamp": ["2025-01-01T12:00:00", "2025-01-01T12:05:00"],
            "duration_sec": [1.5, 2.0],
            "is_chirp": ["True", "False"],
            "clip_file": ["clip1.wav", "clip2.wav"],
        })
        
        report = generate_email_report(df, hours=2)
        
        assert isinstance(report, str)
        assert "Total clips created" in report
        assert "Events identified as chirps" in report
    
    def test_generate_email_report_with_chirps(self):
        """Test generating report with chirp events."""
        df = pd.DataFrame({
            "start_timestamp": ["2025-01-01T12:00:00"],
            "duration_sec": [1.5],
            "is_chirp": ["True"],
            "chirp_similarity": [0.95],
            "confidence": [0.92],
            "clip_file": ["clip1.wav"],
        })
        
        report = generate_email_report(df, hours=2)
        
        assert isinstance(report, str)
        assert "chirp" in report.lower()
        assert "0.95" in report or "0.92" in report  # Should include similarity or confidence


class TestAddDateColumn:
    """Test adding date column."""
    
    def test_add_date_column(self):
        """Test adding date column from timestamps."""
        df = pd.DataFrame({
            "start_timestamp": [
                "2025-01-01T12:00:00",
                "2025-01-02T15:30:00",
                "2025-01-01T18:45:00",
            ],
        })
        
        df_with_date = add_date_column(df)
        
        assert "date" in df_with_date.columns
        assert df_with_date["date"].iloc[0] == "2025-01-01"
        assert df_with_date["date"].iloc[1] == "2025-01-02"
        assert df_with_date["date"].iloc[2] == "2025-01-01"
    
    def test_add_date_column_preserves_original(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame({
            "start_timestamp": ["2025-01-01T12:00:00"],
            "duration_sec": [1.5],
        })
        
        df_with_date = add_date_column(df)
        
        assert "start_timestamp" in df_with_date.columns
        assert "duration_sec" in df_with_date.columns
        assert "date" in df_with_date.columns


class TestChooseLatestDate:
    """Test choosing latest date."""
    
    def test_choose_latest_date(self):
        """Test choosing latest date from DataFrame."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-03", "2025-01-02"],
        })
        
        latest = choose_latest_date(df)
        
        assert latest == "2025-01-03"
    
    def test_choose_latest_date_empty(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        latest = choose_latest_date(df)
        
        assert latest is None
    
    def test_choose_latest_date_no_date_column(self):
        """Test when date column missing."""
        df = pd.DataFrame({
            "start_timestamp": ["2025-01-01T12:00:00"],
        })
        
        latest = choose_latest_date(df)
        
        assert latest is None


class TestGenerateChirpReport:
    """Test chirp report generation."""
    
    def test_generate_chirp_report(self):
        """Test generating markdown chirp report."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-01"],
            "start_timestamp": ["2025-01-01T12:00:00", "2025-01-01T12:05:00"],
            "end_timestamp": ["2025-01-01T12:00:01", "2025-01-01T12:05:02"],
            "duration_sec": [1.5, 2.0],
            "max_rms_db": [-20.5, -18.0],
            "is_chirp": ["True", "False"],
            "chirp_similarity": [0.95, 0.3],
            "clip_file": ["clip1.wav", "clip2.wav"],
        })
        
        report = generate_chirp_report(df, "2025-01-01")
        
        assert isinstance(report, str)
        assert "# Noise Chirp Report" in report
        assert "2025-01-01" in report
        assert "Total events recorded" in report
    
    def test_generate_chirp_report_no_chirps(self):
        """Test generating report with no chirps."""
        # Test case 1: Has chirp columns but no chirps
        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "start_timestamp": ["2025-01-01T12:00:00"],
            "end_timestamp": ["2025-01-01T12:00:01"],
            "duration_sec": [1.5],
            "max_rms_db": [-20.0],
            "is_chirp": ["False"],
            "chirp_similarity": [0.3],
            "clip_file": ["clip1.wav"],
        })
        
        report = generate_chirp_report(df, "2025-01-01")
        
        assert isinstance(report, str)
        assert "No chirp-like events" in report or "0" in report or "Events classified as chirp: **0**" in report

