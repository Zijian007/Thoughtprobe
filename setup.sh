#!/bin/bash
# Setup script for main_leaves_standalone project

set -e  # Exit on error

echo "=========================================="
echo "Main Leaves Standalone - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if we're in the right directory
if [ ! -f "main_leaves.py" ]; then
    echo "Error: main_leaves.py not found. Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created at ./venv"
    echo "To activate it, run: source venv/bin/activate"
    echo ""
fi

# Install dependencies
read -p "Do you want to install dependencies now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if virtual environment should be used
    if [ -d "venv" ]; then
        read -p "Install in virtual environment? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            source venv/bin/activate
        fi
    fi
    
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
    echo ""
fi

# Check for required external resources
echo "=========================================="
echo "Checking for required external resources..."
echo "=========================================="
echo ""

# Function to check if directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo "✓ Found: $1"
        return 0
    else
        echo "✗ Missing: $1"
        return 1
    fi
}

# Check for common paths (adjust as needed)
root_path="/hdd/zijianwang/CoT-decoding"
echo "Checking at root_path: $root_path"
echo ""

missing_resources=0

echo "1. Checking for dataset directory..."
if check_dir "$root_path/dataset"; then
    echo "   Dataset directory found."
else
    echo "   WARNING: Dataset directory not found!"
    echo "   You need dataset files for your chosen dataset (e.g., gsm8k, multiarith, etc.)"
    missing_resources=$((missing_resources + 1))
fi
echo ""

echo "2. Checking for probe weights directory..."
if check_dir "$root_path/Probe_weights"; then
    echo "   Probe weights directory found."
    echo "   Make sure you have weights for your specific model/dataset/prober combination."
else
    echo "   WARNING: Probe weights directory not found!"
    echo "   You need pre-trained prober weights at: $root_path/Probe_weights/{model}/{dataset}/{prober_type}/"
    missing_resources=$((missing_resources + 1))
fi
echo ""

echo "3. Checking for HuggingFace cache directory..."
hf_cache="/hdd/zijianwang/HF_CACHE"
if [ -d "$hf_cache" ]; then
    echo "   ✓ HuggingFace cache directory found at $hf_cache"
else
    echo "   Note: HuggingFace cache directory not found at $hf_cache"
    echo "   Models will be downloaded on first run to default cache location or this path will be created."
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""

if [ $missing_resources -eq 0 ]; then
    echo "✓ All required resources appear to be available!"
    echo ""
    echo "Next steps:"
    echo "  1. Review and modify the configuration in main_leaves.py (lines 275-311)"
    echo "  2. Or use example_run.py as a template for custom configurations"
    echo "  3. Run: python main_leaves.py"
else
    echo "⚠ Some required resources are missing ($missing_resources items)"
    echo ""
    echo "Please ensure you have:"
    echo "  - Dataset files in $root_path/dataset/"
    echo "  - Trained prober weights in $root_path/Probe_weights/"
    echo ""
    echo "You can still run the code, but it will fail if these resources are not available."
fi

echo ""
echo "For detailed information, see README.md"
echo ""
echo "Setup script completed!"

