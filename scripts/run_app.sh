#!/bin/bash

# Fake Reviews Detection App Runner
# This script sets up the environment and runs the Streamlit application

# Move to project root directory
cd "$(dirname "$0")/.."

echo "🚀 Starting Fake Reviews Detection System..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade requirements
echo "📚 Installing/updating dependencies..."
pip install -r requirements.txt

# Download NLTK data if needed
echo "📝 Setting up NLTK data..."
python -c "
import nltk
try:
    nltk.data.find('corpora/stopwords')
    print('✅ NLTK stopwords already downloaded')
except LookupError:
    print('📥 Downloading NLTK stopwords...')
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK stopwords downloaded successfully')
"

# Check if model files exist
echo "🤖 Checking model files..."
if [ ! -d "model_weights" ]; then
    echo "❌ Model weights directory not found!"
    echo "Please make sure you have trained the models first by running the Jupyter notebook."
    exit 1
fi

model_count=$(ls model_weights/*.pkl 2>/dev/null | wc -l)
if [ $model_count -eq 0 ]; then
    echo "❌ No model files found in model_weights directory!"
    echo "Please train the models first by running the Jupyter notebook."
    exit 1
else
    echo "✅ Found $model_count model files"
fi

# Test the application first
echo "🧪 Testing application components..."
cd src
python test_app.py
test_result=$?

if [ $test_result -ne 0 ]; then
    echo "❌ Tests failed! Please check the errors above."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted."
        exit 1
    fi
fi

# Run the Streamlit application from src directory
echo "🌟 Starting Streamlit server..."
echo "🔗 The application will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
