"""
Fake Reviews Detection Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path


from models.model_manager import ModelManager
from utils.text_preprocessing import TextPreprocessor, validate_text_input, extract_text_features, text_process_pipeline
from utils.ui_components import (
    display_prediction_result, 
    display_text_analysis, 
    create_model_comparison_table,
    display_app_info,
    add_custom_css,
    display_model_performance_comparison
)
from config.config import APP_CONFIG, UI_CONFIG

# Make text_process_pipeline available globally for pickle compatibility
import builtins
builtins.text_process_pipeline = text_process_pipeline
globals()['text_process_pipeline'] = text_process_pipeline

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
add_custom_css()

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def main():
    """Main application function"""
    
    # Title and description
    st.title(APP_CONFIG["title"])
    st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 2rem; 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white;'>
            <h3>🤖 AI-Powered Fake Review Detection System</h3>
            <p>Analyze reviews using state-of-the-art machine learning models to detect fake content</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection
    with st.sidebar:
        st.header(APP_CONFIG["sidebar_title"])
        
        # Model selection
        model_manager = st.session_state.model_manager
        available_models = model_manager.get_available_models()
        
        model_category = st.selectbox(
            "🏷️ Select Model Category:",
            options=list(available_models.keys()),
            index=0
        )
        
        model_name = st.selectbox(
            "🤖 Select Model:",
            options=list(available_models[model_category].keys()),
            index=0
        )
        
        # Display model information
        model_info = model_manager.get_model_info(model_category, model_name)
        if model_info:
            st.info(f"**Description:** {model_info['description']}")
        
        # Analysis options
        st.markdown("---")
        st.subheader("⚙️ Analysis Options")
        
        show_text_analysis = st.checkbox("Show Text Analysis", value=True)
        show_confidence_chart = st.checkbox("Show Confidence Chart", value=True)
        show_processing_details = st.checkbox("Show Processing Details", value=False)
        
        # Display app info
        display_app_info()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Review Text")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload Text File"]
        )
        
        review_text = ""
        
        if input_method == "Type/Paste Text":
            review_text = st.text_area(
                "Enter the review text to analyze:",
                height=200,
                placeholder="Paste or type the review text here...",
                max_chars=APP_CONFIG["max_review_length"]
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'csv'],
                help="Upload a .txt file containing the review text"
            )
            
            if uploaded_file:
                try:
                    review_text = uploaded_file.read().decode('utf-8')
                    st.text_area("Uploaded text:", value=review_text, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze Review", 
            type="primary", 
            use_container_width=True
        )
        
        # Sample texts for quick testing
        st.markdown("---")
        st.subheader("🧪 Quick Test Samples")
        
        col_sample1, col_sample2 = st.columns(2)
        
        with col_sample1:
            if st.button("📄 Real Review Sample"):
                review_text = """This product exceeded my expectations! The quality is outstanding and it arrived exactly as described. I've been using it for a few weeks now and it works perfectly. The customer service was also very helpful when I had questions about the features. Would definitely recommend this to anyone looking for a reliable solution."""
                st.rerun()
        
        with col_sample2:
            if st.button("⚠️ Suspicious Review Sample"):
                review_text = """Amazing product!!! Best ever!!! 5 stars!!! Everyone should buy this right now!!! Super fast shipping and perfect quality!!! Will definitely order again soon!!! Highly recommended!!! Great seller!!! Perfect transaction!!! Thank you so much!!!"""
                st.rerun()
    
    with col2:
        # Model comparison table
        create_model_comparison_table(available_models)
        
        # Performance comparison
        if st.button("📊 Show Performance Comparison"):
            st.session_state.show_performance = True
    
    # Analysis results
    if analyze_button and review_text:
        # Validate input
        is_valid, error_message = validate_text_input(review_text, APP_CONFIG["max_review_length"])
        
        if not is_valid:
            st.error(error_message)
            return
        
        # Show processing indicator
        with st.spinner(f"🔄 Analyzing review with {model_name}..."):
            # Make prediction
            result = model_manager.predict(review_text, model_category, model_name)
            
            # Store in history
            st.session_state.prediction_history.append({
                'text': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                'model': f"{model_category} - {model_name}",
                'prediction': result.get('prediction_label', 'Error'),
                'confidence': result.get('confidence', 0.0)
            })
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Analysis Results")
        
        # Main prediction result
        display_prediction_result(result)
        
        # Additional analysis
        if show_text_analysis and result['success']:
            st.markdown("---")
            text_features = extract_text_features(review_text)
            display_text_analysis(review_text, text_features)
        
        # Processing details
        if show_processing_details and result['success']:
            st.markdown("---")
            st.subheader("🔧 Processing Details")
            
            with st.expander("View Processing Details"):
                st.json({
                    'original_text_length': len(review_text),
                    'processed_text_length': len(result.get('processed_text', '')),
                    'model_type': result['model_info'].get('type', ''),
                    'prediction_time': f"{result['prediction_time']:.4f} seconds",
                    'confidence_score': f"{result['confidence']:.4f}"
                })
    
    # Performance comparison section
    if st.session_state.get('show_performance', False):
        st.markdown("---")
        display_model_performance_comparison()
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📚 Recent Predictions")
        
        # Display last 5 predictions
        import pandas as pd
        history_df = pd.DataFrame(st.session_state.prediction_history[-5:])
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; color: #666;'>
            <p>🔍 Fake Reviews Detection System | Built with Streamlit & Scikit-learn</p>
            <p>For educational and research purposes only</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
