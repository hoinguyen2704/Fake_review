"""
UI utility functions for Streamlit app
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any

def display_prediction_result(result: Dict[str, Any]):
    """
    Display prediction results in a formatted way
    Args:
        result (dict): Prediction result dictionary
    """
    if not result['success']:
        st.error(f"❌ {result.get('error', 'Unknown error occurred')}")
        return
    
    # Main prediction result
    prediction_label = result['prediction_label']
    confidence = result['confidence']
    
    if prediction_label == 'Real Review':
        st.success(f"✅ **{prediction_label}** (Confidence: {confidence:.2%})")
    else:
        st.error(f"🚨 **{prediction_label}** (Confidence: {confidence:.2%})")
    
    # Create columns for detailed information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Confidence meter
        if result['probabilities']:
            display_confidence_chart(result['probabilities'])
        
        # Model information
        with st.expander("🔧 Model Information"):
            model_info = result['model_info']
            st.write(f"**Model:** {model_info['name']}")
            st.write(f"**Category:** {model_info['category']}")
            st.write(f"**Type:** {model_info['type']}")
            st.write(f"**Description:** {model_info['description']}")
    
    with col2:
        # Performance metrics
        st.metric("Confidence", f"{confidence:.2%}")
        st.metric("Prediction Time", f"{result['prediction_time']:.3f}s")
        
        # Show processed text sample
        with st.expander("📝 Processed Text Sample"):
            st.text(result['processed_text'])

def display_confidence_chart(probabilities: Dict[str, float]):
    """
    Display confidence chart using plotly
    Args:
        probabilities (dict): Dictionary of class probabilities
    """
    if not probabilities or len(probabilities) < 2:
        return
    
    # Create gauge chart for confidence
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Map labels to more readable format
    readable_labels = []
    colors = []
    for label in labels:
        if 'OR' in str(label) or 'Real' in str(label):
            readable_labels.append('Real Review')
            colors.append('#28a745')  # Green
        elif 'CG' in str(label) or 'Fake' in str(label):
            readable_labels.append('Fake Review')
            colors.append('#dc3545')  # Red
        else:
            readable_labels.append(str(label))
            colors.append('#6c757d')  # Gray
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=readable_labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2%}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Review Type",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_text_analysis(text: str, features: Dict[str, Any]):
    """
    Display text analysis metrics
    Args:
        text (str): Original text
        features (dict): Text features dictionary
    """
    st.subheader("📊 Text Analysis")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Characters", features.get('char_count', 0))
        st.metric("Words", features.get('word_count', 0))
    
    with col2:
        st.metric("Sentences", features.get('sentence_count', 0))
        st.metric("Avg Word Length", f"{features.get('avg_word_length', 0):.1f}")
    
    with col3:
        st.metric("Uppercase Ratio", f"{features.get('uppercase_ratio', 0):.1%}")
        st.metric("Punctuation Ratio", f"{features.get('punctuation_ratio', 0):.1%}")
    
    with col4:
        # Text complexity indicator
        word_count = features.get('word_count', 0)
        if word_count < 10:
            complexity = "Very Short"
            color = "orange"
        elif word_count < 50:
            complexity = "Short"
            color = "yellow"
        elif word_count < 200:
            complexity = "Medium"
            color = "green"
        else:
            complexity = "Long"
            color = "blue"
        
        st.metric("Text Length", complexity)

def create_model_comparison_table(models_info: Dict[str, Any]):
    """
    Create a comparison table of available models
    Args:
        models_info (dict): Dictionary containing model information
    """
    st.subheader("🤖 Available Models")
    
    data = []
    for category, models in models_info.items():
        for model_name, model_config in models.items():
            data.append({
                'Category': category,
                'Model': model_name,
                'Type': model_config['type'].upper(),
                'Description': model_config['description']
            })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def display_app_info():
    """Display application information and instructions"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This application uses machine learning models to detect fake reviews.
        
        **How to use:**
        1. Select a model category and specific model
        2. Enter or paste a review text
        3. Click 'Analyze Review' to get prediction
        
        **Model Types:**
        - **BoW**: Bag of Words with various classifiers
        - **Pipeline**: Complete preprocessing + classification pipeline
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Features")
    st.sidebar.markdown(
        """
        - Real-time fake review detection
        - Multiple ML algorithms
        - Confidence scores and probabilities
        - Text analysis metrics
        - Model performance comparison
        - Interactive visualizations
        """
    )

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .stAlert > div {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        .metric-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
        }
        
        .prediction-success {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
            font-weight: bold;
        }
        
        .prediction-warning {
            background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
            font-weight: bold;
        }
        
        .model-info {
            background-color: #f8f9fa;
            padding: 1rem;
            border-left: 4px solid #007bff;
            border-radius: 0.25rem;
        }
        </style>
    """, unsafe_allow_html=True)

def display_model_performance_comparison():
    """Display a comparison of model performances (placeholder)"""
    st.subheader("📈 Model Performance Comparison")
    
    # This would typically come from your training results
    # For demo purposes, using sample data
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Decision Tree', 'KNN'],
        'Accuracy': [0.92, 0.89, 0.88, 0.85, 0.87],
        'Training Time (s)': [2.1, 15.3, 0.8, 3.2, 0.2],
        'Inference Time (ms)': [1.2, 5.8, 0.9, 1.5, 8.3]
    }
    
    df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig_acc = px.bar(df, x='Model', y='Accuracy', 
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='viridis')
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Speed comparison
        fig_speed = px.scatter(df, x='Training Time (s)', y='Inference Time (ms)',
                              size='Accuracy', hover_name='Model',
                              title='Speed vs Accuracy Trade-off')
        fig_speed.update_layout(height=400)
        st.plotly_chart(fig_speed, use_container_width=True)
