#!/bin/bash
"""
Quick test script for GPD Julia computation.

This script runs a quick test to verify everything works.
"""

echo "Running GPD Julia Test"
echo "======================"

# Check if Julia is available
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed"
    exit 1
fi

# Get thread count from command line or use default
THREADS=${1:-4}
echo "Using $THREADS threads for testing"

# Run test
echo "Running test computation (10 parameter combinations)..."
julia --threads=$THREADS main.jl --test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo "You can now run the full computation with:"
    echo "  julia --threads=$THREADS main.jl"
    echo "  julia --threads=20 main.jl  # for maximum performance"
else
    echo ""
    echo "❌ Test failed"
    exit 1
fi
