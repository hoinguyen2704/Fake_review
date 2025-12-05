#!/bin/bash

# Demo Script for Fake Reviews Detection
# This script runs the interactive demo from the root directory

echo "🎮 FAKE REVIEWS DETECTION - INTERACTIVE DEMO"
echo "==========================================="

# Move to project root directory
cd "$(dirname "$0")/.."

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/demo.py" ]; then
    echo "❌ Error: Could not find project structure"
    echo "   Make sure 'src/demo.py' exists"
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

# Download NLTK data if needed
echo "📝 Setting up NLTK data..."
python -c "
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print('📥 Downloading NLTK stopwords...')
    nltk.download('stopwords', quiet=True)
"

# Run demo
echo "🚀 Starting interactive demo..."
cd src
python demo.py
