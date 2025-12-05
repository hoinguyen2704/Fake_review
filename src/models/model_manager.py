"""
Model management and inference utilities
"""
import pickle
import time
import logging
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Any, Optional
import sys
import builtins

# Import các module cần thiết
from config.config import AVAILABLE_MODELS
from utils.text_preprocessing import TextPreprocessor, text_process_pipeline

# Import naive_bayes module for model compatibility
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import naive_bayes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make text_process_pipeline available globally for pickle compatibility
sys.modules[__name__].text_process_pipeline = text_process_pipeline
builtins.text_process_pipeline = text_process_pipeline

# Also make it available in __main__ module if running as script
if hasattr(sys.modules.get('__main__'), '__file__'):
    sys.modules['__main__'].text_process_pipeline = text_process_pipeline

class ModelManager:
    """
    Handles loading and inference of trained models
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.loaded_vectorizers = {}
        self.text_preprocessor = TextPreprocessor()
        
    def load_model(self, model_category: str, model_name: str) -> bool:
        """
        Load a specific model and its components
        Args:
            model_category (str): Category of the model
            model_name (str): Name of the model
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_key = f"{model_category}_{model_name}"
            
            if model_key in self.loaded_models:
                logger.info(f"Model {model_key} already loaded")
                return True
            
            # Kiểm tra xem model có tồn tại trong config không
            if model_category not in AVAILABLE_MODELS:
                logger.error(f"Model category '{model_category}' not found in config")
                return False
                
            if model_name not in AVAILABLE_MODELS[model_category]:
                logger.error(f"Model '{model_name}' not found in category '{model_category}'")
                return False
            
            model_config = AVAILABLE_MODELS[model_category][model_name]
            model_path = Path(model_config["path"])
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the main model
            # Ensure text_process_pipeline is available globally for pipeline models
            if model_config.get('type') == 'pipeline':
                # Make function available in multiple namespaces for pickle compatibility
                builtins.text_process_pipeline = text_process_pipeline
                sys.modules['__main__'].text_process_pipeline = text_process_pipeline
                
                # Also try to set it in the main module's globals
                if hasattr(sys.modules.get('__main__'), '__dict__'):
                    sys.modules['__main__'].__dict__['text_process_pipeline'] = text_process_pipeline
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.loaded_models[model_key] = {
                'model': model,
                'config': model_config,
                'type': model_config.get('type', 'unknown')
            }
            
            # Load vectorizer for BoW models
            if model_config.get('type') == 'bow' and 'vectorizer' in model_config:
                vectorizer_path = Path(model_config['vectorizer'])
                if vectorizer_path.exists():
                    try:
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        self.loaded_vectorizers[model_key] = vectorizer
                        logger.info(f"Vectorizer loaded for {model_key}")
                    except Exception as e:
                        logger.warning(f"Could not load vectorizer for {model_key}: {str(e)}")
                        # Không return False vì model vẫn có thể hoạt động
                else:
                    logger.warning(f"Vectorizer not found: {vectorizer_path}")
                    # Không return False vì model vẫn có thể hoạt động
            
            logger.info(f"Successfully loaded model: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_category}_{model_name}: {str(e)}")
            return False
    
    def predict(self, text: str, model_category: str, model_name: str) -> Dict[str, Any]:
        """
        Make prediction using specified model
        Args:
            text (str): Input text to classify
            model_category (str): Category of the model
            model_name (str): Name of the model
        Returns:
            dict: Prediction results
        """
        model_key = f"{model_category}_{model_name}"
        
        # Load model if not already loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_category, model_name):
                return self._create_error_result("Failed to load model")
        
        try:
            start_time = time.time()
            model_info = self.loaded_models[model_key]
            model = model_info['model']
            model_type = model_info.get('type', 'unknown')
            
            # Preprocess text based on model type
            if model_type == 'bow':
                processed_text = self.text_preprocessor.preprocess_for_bow(text)
                
                # Kiểm tra xem có vectorizer không
                if model_key in self.loaded_vectorizers:
                    vectorizer = self.loaded_vectorizers[model_key]
                    # Transform text to feature vector
                    try:
                        text_vector = vectorizer.transform([processed_text])
                        # Chuyển đổi sang numpy array nếu cần
                        if hasattr(text_vector, 'toarray'):
                            text_vector = text_vector.toarray()
                        elif hasattr(text_vector, 'numpy'):
                            text_vector = text_vector.numpy()
                    except Exception as e:
                        logger.error(f"Error transforming text with vectorizer: {str(e)}")
                        return self._create_error_result(f"Vectorizer transformation failed: {str(e)}")
                else:
                    logger.warning(f"No vectorizer found for {model_key}, trying direct prediction")
                    # Thử predict trực tiếp với text đã xử lý
                    text_vector = [processed_text]
                
            elif model_type == 'pipeline':
                # Pipeline models handle preprocessing internally
                processed_text = self.text_preprocessor.preprocess_for_pipeline(text)
                text_vector = [processed_text]
            
            elif model_type == 'cnn':
                # CNN models cần tokenization và padding
                try:
                    from tensorflow.keras.preprocessing.text import Tokenizer
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    
                    # Load tokenizer nếu có
                    tokenizer_path = model_info['config'].get('tokenizer')
                    if tokenizer_path and Path(tokenizer_path).exists():
                        with open(tokenizer_path, 'rb') as f:
                            tokenizer = pickle.load(f)
                        
                        # Tokenize và pad text
                        sequences = tokenizer.texts_to_sequences([processed_text])
                        text_vector = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
                    else:
                        logger.warning("No tokenizer found for CNN model")
                        text_vector = [processed_text]
                        
                except ImportError:
                    logger.error("TensorFlow not available for CNN prediction")
                    return self._create_error_result("TensorFlow required for CNN models")
                except Exception as e:
                    logger.error(f"Error processing text for CNN: {str(e)}")
                    return self._create_error_result(f"CNN text processing failed: {str(e)}")
            
            else:
                # Fallback: treat as general text input
                processed_text = self.text_preprocessor.preprocess_for_pipeline(text)
                text_vector = [processed_text]
            
            # Make prediction
            try:
                if hasattr(model, 'predict'):
                    prediction = model.predict(text_vector)
                    # Lấy prediction đầu tiên nếu là array
                    if isinstance(prediction, (np.ndarray, list)):
                        prediction = prediction[0]
                else:
                    return self._create_error_result("Model does not have predict method")
                
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
                return self._create_error_result(f"Model prediction failed: {str(e)}")
            
            # Get prediction probabilities if available
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(text_vector)
                    if isinstance(probabilities, (np.ndarray, list)):
                        probabilities = probabilities[0]
                    confidence = float(np.max(probabilities))
                    
                    # Get class labels
                    if hasattr(model, 'classes_'):
                        classes = model.classes_
                        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
                    else:
                        # Fallback cho binary classification
                        prob_dict = {'OR (Real)': float(probabilities[0]), 'CG (Fake)': float(probabilities[1])}
                        
                elif hasattr(model, 'decision_function'):
                    # For SVM
                    decision = model.decision_function(text_vector)
                    if isinstance(decision, (np.ndarray, list)):
                        decision = decision[0]
                    confidence = abs(float(decision))
                    prob_dict = {'confidence_score': confidence}
                else:
                    confidence = 0.5
                    prob_dict = {}
                    
            except Exception as e:
                logger.warning(f"Could not get probabilities: {str(e)}")
                confidence = 0.5
                prob_dict = {}
            
            prediction_time = time.time() - start_time
            
            # Format result
            result = {
                'prediction': str(prediction),
                'prediction_label': self._get_prediction_label(str(prediction)),
                'confidence': confidence,
                'probabilities': prob_dict,
                'prediction_time': prediction_time,
                'model_info': {
                    'name': model_name,
                    'category': model_category,
                    'type': model_type,
                    'description': model_info['config'].get('description', 'No description available')
                },
                'processed_text': processed_text[:100] + "..." if len(processed_text) > 100 else processed_text,
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return self._create_error_result(f"Prediction failed: {str(e)}")
    
    def _get_prediction_label(self, prediction: str) -> str:
        """Convert prediction to human-readable label"""
        prediction_str = str(prediction).upper()
        
        # Mapping các giá trị prediction phổ biến
        label_mapping = {
            'OR': 'Real Review',
            'CG': 'Fake Review',
            '0': 'Real Review',
            '1': 'Fake Review',
            'REAL': 'Real Review',
            'FAKE': 'Fake Review',
            'TRUE': 'Real Review',
            'FALSE': 'Fake Review'
        }
        
        return label_mapping.get(prediction_str, f'Unknown ({prediction})')
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'prediction': 'Error',
            'prediction_label': 'Error',
            'confidence': 0.0,
            'probabilities': {},
            'prediction_time': 0.0,
            'model_info': {},
            'processed_text': '',
            'success': False,
            'error': error_message
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return AVAILABLE_MODELS
    
    def get_model_info(self, model_category: str, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            return AVAILABLE_MODELS[model_category][model_name]
        except KeyError:
            return {}
    
    def unload_model(self, model_category: str, model_name: str) -> bool:
        """Unload a specific model from memory"""
        model_key = f"{model_category}_{model_name}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            
        if model_key in self.loaded_vectorizers:
            del self.loaded_vectorizers[model_key]
            
        logger.info(f"Unloaded model: {model_key}")
        return True
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def reload_model(self, model_category: str, model_name: str) -> bool:
        """Reload a specific model"""
        # Unload first
        self.unload_model(model_category, model_name)
        # Load again
        return self.load_model(model_category, model_name)
    
    def get_model_status(self, model_category: str, model_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific model"""
        model_key = f"{model_category}_{model_name}"
        
        if model_key not in self.loaded_models:
            return {
                'loaded': False,
                'status': 'Not loaded',
                'error': 'Model not loaded'
            }
        
        model_info = self.loaded_models[model_key]
        
        return {
            'loaded': True,
            'status': 'Ready',
            'type': model_info.get('type', 'unknown'),
            'has_vectorizer': model_key in self.loaded_vectorizers,
            'config': model_info.get('config', {}),
            'model_class': type(model_info['model']).__name__
        }