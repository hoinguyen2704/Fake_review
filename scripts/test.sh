#!/bin/bash

# Quick Test Script for Fake Reviews Detection
# This script runs tests from the root directory

echo "🧪 FAKE REVIEWS DETECTION - QUICK TEST"
echo "======================================"

# Move to project root directory
cd "$(dirname "$0")/.."

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/test_app.py" ]; then
    echo "❌ Error: Could not find project structure"
    echo "   Make sure 'src/test_app.py' exists"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements if needed
if [ ! -f ".venv/installed" ]; then
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
    touch .venv/installed
fi

# Run tests
echo "🚀 Running application tests..."
cd src
python test_app.py
test_result=$?

# Return to root
cd ..

if [ $test_result -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo "💡 You can now run the app with: ./start_app.sh"
else
    echo ""
    echo "❌ Some tests failed. Please check the output above."
fi

exit $test_result
