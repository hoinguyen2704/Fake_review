#!/usr/bin/env python3
"""
Fake Reviews Detection - Interactive Demo
This script demonstrates the core functionality of the application programmatically.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from models.model_manager import ModelManager
from utils.text_preprocessing import TextPreprocessor
from config.config import AVAILABLE_MODELS

def demo_prediction(text, model_category="Bag of Words Models", model_name="Naive Bayes"):
    """Demonstrate a single prediction"""
    print(f"\n🔍 ANALYZING: '{text[:60]}{'...' if len(text) > 60 else ''}'")
    print("=" * 80)
    
    # Initialize components
    model_manager = ModelManager()
    preprocessor = TextPreprocessor()
    
    # Preprocess text
    processed_text = preprocessor.preprocess_text(text)
    print(f"📝 Processed Text: '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")
    
    # Get prediction
    start_time = time.time()
    result = model_manager.predict(text, model_category, model_name)
    inference_time = time.time() - start_time
    
    # Display results
    confidence_emoji = "🔴" if result.get('confidence', 0) > 80 else "🟡" if result.get('confidence', 0) > 60 else "🟢"
    
    print(f"🤖 Model: {model_category} - {model_name}")
    print(f"{confidence_emoji} Prediction: {result.get('prediction_label', 'Error')}")
    print(f"🎯 Confidence: {result.get('confidence', 0.0):.2f}%")
    print(f"⏱️ Inference Time: {inference_time:.4f}s")
    
    return result

def demo_multiple_models(text, models_to_test):
    """Test the same text with multiple models"""
    print(f"\n{'='*90}")
    print(f"🔍 MULTI-MODEL ANALYSIS")
    print(f"💬 Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
    print(f"{'='*90}")
    
    results = []
    for model_tuple in models_to_test:
        model_category, model_name = model_tuple
        try:
            start_time = time.time()
            model_manager = ModelManager()
            result = model_manager.predict(text, model_category, model_name)
            inference_time = time.time() - start_time
            
            confidence = result.get('confidence', 0.0)
            confidence_emoji = "🔴" if confidence > 80 else "🟡" if confidence > 60 else "🟢"
            
            print(f"{confidence_emoji} {model_name:20} | {result.get('prediction_label', 'Error'):12} | {confidence:6.1f}% | {inference_time:.3f}s")
            results.append((model_category, model_name, result, inference_time))
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {str(e)[:50]}...")
    
    return results

def interactive_demo():
    """Interactive demo where user can input their own text"""
    print("\n🎮 INTERACTIVE DEMO MODE")
    print("=" * 50)
    print("Enter your own review text to analyze (or 'quit' to exit)")
    
    # Available models
    all_models = []
    for category, models in AVAILABLE_MODELS.items():
        for model_name in models.keys():
            all_models.append((category, model_name))
    
    print(f"\nAvailable models: {len(all_models)} total")
    for i, (category, model_name) in enumerate(all_models, 1):
        print(f"  {i}. {category} - {model_name}")
    
    while True:
        try:
            print("\n" + "-" * 50)
            user_text = input("\n💬 Enter review text: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_text:
                print("❌ Please enter some text.")
                continue
            
            # Ask for model selection
            print(f"\nSelect model (1-{len(all_models)}) or press Enter for default:")
            model_choice = input("Model number: ").strip()
            
            if model_choice:
                try:
                    model_index = int(model_choice) - 1
                    if 0 <= model_index < len(all_models):
                        selected_model = all_models[model_index]
                    else:
                        print("❌ Invalid model number, using default.")
                        selected_model = all_models[0]
                except ValueError:
                    print("❌ Invalid input, using default model.")
                    selected_model = all_models[0]
            else:
                selected_model = all_models[0]
            
            # Perform prediction
            category, name = selected_model
            demo_prediction(user_text, category, name)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main demo function"""
    print("🚀 FAKE REVIEWS DETECTION - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Sample reviews for testing
    sample_reviews = [
        {
            "text": "This product is absolutely amazing! I love it so much and would definitely recommend it to everyone!",
            "description": "Potentially fake (overly enthusiastic)"
        },
        {
            "text": "The item arrived quickly and works as expected. Good quality for the price.",
            "description": "Likely genuine (balanced, specific)"
        },
        {
            "text": "BEST PRODUCT EVER!!! 5 STARS!!! Everyone should buy this RIGHT NOW!!!",
            "description": "Likely fake (excessive caps, exclamation)"
        },
        {
            "text": "Decent product. It does what it's supposed to do. Shipping was fast.",
            "description": "Likely genuine (objective, factual)"
        },
        {
            "text": "OMG this is the most incredible thing I've ever bought! Perfect in every way!",
            "description": "Potentially fake (extreme language)"
        }
    ]
    
    # Test different models
    models_to_test = [
        ("Bag of Words Models", "Naive Bayes"),
        ("Bag of Words Models", "Logistic Regression"),
        ("Bag of Words Models", "Support Vector Machine"),
        ("Pipeline Models", "TF-IDF + Logistic Regression"),
        ("Pipeline Models", "TF-IDF + Support Vector Machine")
    ]
    
    print(f"\n📋 Testing {len(sample_reviews)} sample reviews with {len(models_to_test)} models...")
    
    # Quick demo mode
    print(f"\n🎯 QUICK DEMO MODE")
    print("=" * 50)
    
    for i, review_data in enumerate(sample_reviews, 1):
        print(f"\n📝 SAMPLE {i}: {review_data['description']}")
        demo_multiple_models(review_data['text'], models_to_test[:3])  # Test with first 3 models
        
        if i < len(sample_reviews):
            response = input(f"\n⏭️ Continue to next sample? (y/n/i for interactive): ").strip().lower()
            if response == 'n':
                break
            elif response == 'i':
                interactive_demo()
                break
    
    # Ask if user wants interactive mode
    if input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower() == 'y':
        interactive_demo()
    
    print("\n✅ Demo completed successfully!")
    print("💡 To use the full web interface, run: streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
