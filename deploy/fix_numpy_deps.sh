#!/bin/bash
# Fix NumPy OpenBLAS dependency issue on Raspberry Pi
# Run this script on the Pi if you see "libopenblas.so.0: cannot open shared object file"

set -e

echo "Fixing NumPy OpenBLAS dependency issue..."
echo ""

# Check if we're on a Debian-based system
if ! command -v apt-get &> /dev/null; then
    echo "ERROR: This script requires apt-get (Debian/Ubuntu/Raspberry Pi OS)"
    exit 1
fi

echo "Step 1: Installing OpenBLAS system library..."
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev

echo ""
echo "Step 2: Checking current NumPy installation..."
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy can be imported successfully"
    python3 -c "import numpy; print(f'  NumPy version: {numpy.__version__}')"
    exit 0
fi

echo ""
echo "Step 3: NumPy still not working, trying to fix..."
echo "Option A: Install system NumPy package (recommended)"
sudo apt-get install -y python3-numpy python3-pandas || {
    echo "System packages failed, trying Option B..."
}

echo ""
echo "Option B: Reinstall NumPy via pip with system libraries"
python3 -m pip uninstall -y numpy || true
python3 -m pip install --no-cache-dir "numpy>=1.24.0,<2.1.0"

echo ""
echo "Step 4: Verifying fix..."
if python3 -c "import numpy; import pandas" 2>/dev/null; then
    echo "✓ SUCCESS: NumPy and pandas can now be imported"
    python3 -c "import numpy; import pandas; print(f'  NumPy: {numpy.__version__}'); print(f'  Pandas: {pandas.__version__}')"
else
    echo "✗ ERROR: NumPy still cannot be imported"
    echo ""
    echo "Try manually:"
    echo "  sudo apt-get install -y python3-numpy python3-pandas"
    echo "  python3 -c 'import numpy; print(numpy.__version__)'"
    exit 1
fi

echo ""
echo "✓ Fix complete! The service should now start correctly."

