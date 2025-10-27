#!/bin/bash
"""
Setup script for GPD Julia computation.

This script sets up the Julia environment and installs dependencies.
"""

echo "Setting up GPD Julia Computation Environment"
echo "=========================================="

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed"
    echo "Please install Julia first:"
    echo "  macOS: brew install julia"
    echo "  Linux: sudo apt install julia"
    exit 1
fi

echo "Julia version:"
julia --version

# Install dependencies
echo ""
echo "Installing Julia dependencies..."
julia --project=. -e "using Pkg; Pkg.instantiate()"

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Test the installation
echo ""
echo "Testing installation..."
julia --project=. -e "using QuadGK, SpecialFunctions, CSV, DataFrames, Distributed, ProgressMeter; println(\"✅ All packages loaded successfully\")"

if [ $? -eq 0 ]; then
    echo "✅ Installation test passed"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "Setup complete! You can now run:"
echo "  julia main.jl --test    # Test run (10 combinations)"
echo "  julia main.jl           # Full computation (10,000 combinations)"
