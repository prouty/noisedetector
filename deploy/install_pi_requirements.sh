#!/bin/bash
# Install requirements on Raspberry Pi with proper order and error handling
#
# Note: You may see "WARNING: Error parsing dependencies of pyzmq: Invalid version: 'cpython'"
# This is a harmless warning from pip's dependency resolver and can be safely ignored.

set -e  # Exit on error

echo "Installing Python dependencies on Raspberry Pi..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Method 1: Try system packages first (faster, pre-built)
echo ""
echo "Attempting to install via system packages (recommended for Pi)..."
if command -v apt-get &> /dev/null; then
    echo "Installing system packages..."
    sudo apt-get update
    sudo apt-get install -y python3-numpy python3-pandas python3-dateutil || {
        echo "System packages not available or failed, falling back to pip..."
    }
    
    # If system packages installed successfully, skip pip installation
    if python3 -c "import numpy; import pandas" 2>/dev/null; then
        echo "✓ System packages work correctly, skipping pip installation"
        echo ""
        echo "Verifying installation..."
        python3 -c "import numpy; import pandas; print(f'✓ numpy {numpy.__version__}'); print(f'✓ pandas {pandas.__version__}')"
        echo ""
        echo "✓ All dependencies installed successfully!"
        exit 0
    fi
fi

# Install OpenBLAS library (required for pip-installed numpy)
echo ""
echo "Installing OpenBLAS system library (required for numpy)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y libopenblas0 libopenblas-dev || {
        echo "WARNING: Failed to install OpenBLAS, numpy may not work correctly"
    }
fi

# Method 2: Install via pip with proper order
echo ""
echo "Installing/upgrading pip and build tools..."
# Note: pyzmq warning is harmless - it's a dependency parsing issue, not an error
python3 -m pip install --upgrade pip setuptools wheel

echo ""
echo "Installing numpy first (required for pandas build)..."
# Note: pyzmq warning is harmless - it's a dependency parsing issue, not an error
python3 -m pip install --no-cache-dir "numpy>=1.24.0,<2.1.0" || {
    echo "ERROR: Failed to install numpy"
    echo "Try: sudo apt-get install python3-numpy"
    exit 1
}

echo ""
echo "Installing pandas (this may take a while on Pi if building from source)..."
# Note: pyzmq warning is harmless - it's a dependency parsing issue, not an error
python3 -m pip install --no-cache-dir "pandas>=2.0.0,<2.2.0" || {
    echo "WARNING: pandas installation failed"
    echo "Trying older version with better ARM support..."
    python3 -m pip install --no-cache-dir "pandas>=1.5.0,<2.0.0" || {
        echo "ERROR: Failed to install pandas"
        echo "Try: sudo apt-get install python3-pandas"
        exit 1
    }
}

echo ""
echo "Installing remaining dependencies..."
# Note: pyzmq warning is harmless - it's a dependency parsing issue, not an error
python3 -m pip install --no-cache-dir \
    "python-dateutil>=2.8.0" \
    "pytz>=2024.1" \
    "six>=1.16.0" \
    "tzdata>=2024.1" || {
    echo "ERROR: Failed to install some dependencies"
    exit 1
}

echo ""
echo "Verifying installation..."
python3 -c "import numpy; import pandas; print(f'✓ numpy {numpy.__version__}'); print(f'✓ pandas {pandas.__version__}')" || {
    echo "ERROR: Import test failed"
    exit 1
}

echo ""
echo "✓ All dependencies installed successfully!"

