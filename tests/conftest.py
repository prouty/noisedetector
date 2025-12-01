"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config_loader
import pytest


@pytest.fixture
def project_root_path():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config(project_root_path):
    """Load configuration for testing."""
    config_path = project_root_path / "config.json"
    return config_loader.load_config(config_path if config_path.exists() else None)


@pytest.fixture
def data_dir(project_root_path):
    """Return the data directory path."""
    return project_root_path / "data"

