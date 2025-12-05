"""
Configuration file for Fake Reviews Detection App
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "model_weights"

# Available models configuration
AVAILABLE_MODELS = {
    "Bag of Words Models": {
        "Logistic Regression": {
            "path": MODEL_DIR / "bow_logistic_regression_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Linear model with good interpretability and fast inference"
        },
        "Naive Bayes": {
            "path": MODEL_DIR / "bow_naive_bayes_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Probabilistic classifier, very fast training and inference"
        },
        "Random Forest": {
            "path": MODEL_DIR / "bow_random_forest_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Ensemble learning model, good performance and interpretability"
        }
    },
    "Pipeline Models": {
        "TF-IDF + Logistic Regression": {
            "path": MODEL_DIR / "pipeline_logistic_regression_model.pkl",
            "type": "pipeline",
            "description": "Complete pipeline with TF-IDF transformation and Logistic Regression"
        }
    }
}

# App configuration
APP_CONFIG = {
    "title": "🔍 Fake Reviews Detection System",
    "sidebar_title": "Model Selection & Settings",
    "max_review_length": 5000,
    "default_model_category": "Bag of Words Models",
    "default_model": "Logistic Regression"
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "max_features": 5000,
    "remove_stopwords": True,
    "apply_stemming": True,
    "min_word_length": 2
}

# UI configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730"
    },
    "layout": {
        "sidebar_width": 300,
        "main_width": 700
    }
}

# Performance metrics display
METRICS_CONFIG = {
    "display_metrics": ["confidence", "prediction_time", "model_info"],
    "confidence_threshold": 0.7,
    "show_probability_distribution": True
}
# Tham số xử lý văn bản
TEXT_PROCESSING_PARAMS = {
    'min_word_length': 2,        # Độ dài tối thiểu của từ
    'max_features': 1000,        # Số features tối đa
    'use_nltk_stopwords': True,  # Có sử dụng NLTK stopwords không
    'fallback_stopwords': {      # Stopwords fallback
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
        'should', 'may', 'might', 'must', 'can', 'shall'
    },
    'default_ngram_range': (1, 1),  # N-gram range mặc định
    'default_min_df': 1,            # Document frequency tối thiểu mặc định
    'default_max_df': 1.0,          # Document frequency tối đa mặc định
    'default_analyzer': 'word',     # Analyzer mặc định
    'default_token_pattern': r"(?u)\b\w\w+\b"  # Regex pattern mặc định
}