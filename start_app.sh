#!/bin/bash

# Start Streamlit application
echo "🌟 Starting Streamlit web application..."
echo "=========================================="
echo "🔗 Application will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo "=========================================="
echo ""

cd src
streamlit run app.py
