# CNN Fake Reviews Detection - Preprocessing & Training
# File này tập trung vào việc tiền xử lý dữ liệu và huấn luyện mô hình CNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pickle
import os
import time
from collections import Counter

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Deep Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Settings
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style("whitegrid")
plt.style.use('default')

print("✅ All libraries imported successfully!")

def load_and_explore_data():
    """Load và explore dataset"""
    print("📊 Loading and exploring dataset...")
    
    # Load the dataset
    dataset_path = "Preprocessed Fake Reviews Detection Dataset.csv"
    df = pd.read_csv(dataset_path)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Basic information
    print("\nDataset Information:")
    print("=" * 50)
    print(df.info())
    
    # Missing values analysis
    print("\nMissing Values Analysis:")
    print("=" * 30)
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    return df

def clean_data(df):
    """Clean và preprocess data"""
    print("🧹 Cleaning dataset...")
    
    # Remove unnamed columns if exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        print("✓ Removed 'Unnamed: 0' column")
    
    # Remove any remaining missing values
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"✓ Removed {initial_shape[0] - df.shape[0]} rows with missing values")
    
    # Add text length feature for analysis
    df['text_length'] = df['text_'].apply(len)
    print("✓ Added text_length feature")
    
    print("\nData cleaning completed!")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def analyze_labels(df):
    """Analyze label distribution"""
    print("🏷️ Analyzing label distribution...")
    
    label_counts = df['label'].value_counts()
    print("Label Distribution:")
    print("=" * 20)
    print(label_counts)
    print(f"\nLabel proportions:")
    print(df['label'].value_counts(normalize=True).round(4))
    
    # Visualize label distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    sns.countplot(data=df, x='label', palette='Set2', ax=ax1)
    ax1.set_title('Distribution of Review Labels', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Label', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    
    # Add count labels on bars
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = ['#FF9999', '#66B3FF']
    wedges, texts, autotexts = ax2.pie(label_counts.values, labels=label_counts.index, 
                                      colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    return df

def preprocess_text_for_cnn(df):
    """Text preprocessing cho CNN"""
    print("🧹 Preprocessing text data for CNN...")
    
    def preprocess_text(text):
        """
        Preprocess text cho CNN:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        4. Basic cleaning
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    # Apply preprocessing
    df['text_cleaned'] = df['text_'].apply(preprocess_text)
    print("✓ Text preprocessing completed!")
    
    # Show examples
    print("\n📝 Text preprocessing examples:")
    for i in range(3):
        print(f"\nOriginal {i+1}:")
        print(df['text_'].iloc[i][:100] + "...")
        print(f"\nCleaned {i+1}:")
        print(df['text_cleaned'].iloc[i][:100] + "...")
    
    return df

def analyze_text_length(df):
    """Analyze text length distribution"""
    print("📊 Text Length Analysis:")
    print("=" * 30)
    
    df['cleaned_text_length'] = df['text_cleaned'].apply(len)
    
    print(f"Original text length - Mean: {df['text_length'].mean():.1f}, Max: {df['text_length'].max()}")
    print(f"Cleaned text length - Mean: {df['cleaned_text_length'].mean():.1f}, Max: {df['cleaned_text_length'].max()}")
    
    # Visualize text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original text length
    axes[0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Original Text Length Distribution', fontweight='bold')
    axes[0].set_xlabel('Text Length')
    axes[0].set_ylabel('Frequency')
    
    # Cleaned text length
    axes[1].hist(df['cleaned_text_length'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Cleaned Text Length Distribution', fontweight='bold')
    axes[1].set_xlabel('Text Length')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df

def prepare_sequences_for_cnn(df):
    """Prepare sequences cho CNN"""
    print("🔤 Preparing sequences for CNN...")
    
    # CNN Parameters
    MAX_WORDS = 10000  # Vocabulary size
    MAX_LEN = 200      # Maximum sequence length
    
    print(f"⚙️ CNN Parameters:")
    print(f"• Max words: {MAX_WORDS:,}")
    print(f"• Max sequence length: {MAX_LEN}")
    
    # Tokenization
    print("\n🔤 Tokenizing text data...")
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text_cleaned'])
    
    print(f"✓ Vocabulary size: {len(tokenizer.word_index):,}")
    print(f"✓ Total words: {tokenizer.word_counts.most_common(1)[0][1]:,}")
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['text_cleaned'])
    print(f"✓ Converted {len(sequences)} texts to sequences")
    
    # Pad sequences
    X_padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    print(f"✓ Padded sequences shape: {X_padded.shape}")
    
    # Show examples
    print("\n📝 Tokenization examples:")
    for i in range(2):
        print(f"\nText {i+1}:")
        print(df['text_cleaned'].iloc[i][:100] + "...")
        print(f"\nSequence {i+1}:")
        print(sequences[i][:20])
        print(f"\nPadded sequence {i+1}:")
        print(X_padded[i][:20])
    
    return X_padded, tokenizer, MAX_WORDS, MAX_LEN

def prepare_labels(df):
    """Prepare labels cho CNN"""
    print("🏷️ Preparing labels...")
    
    # Create label mapping
    label_map = {'OR': 0, 'CG': 1}  # OR: Original, CG: Computer Generated
    y = df['label'].map(label_map).values
    
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Label distribution: {np.bincount(y)}")
    print(f"✓ Label mapping: {label_map}")
    
    return y, label_map

def split_data(X_padded, y):
    """Split data thành training và test sets"""
    print("✂️ Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split completed:")
    print(f"• Training set: {X_train.shape}")
    print(f"• Test set: {X_test.shape}")
    print(f"• Training labels: {np.bincount(y_train)}")
    print(f"• Test labels: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def build_cnn_model(vocab_size, max_len, embedding_dim, num_classes=2):
    """Xây dựng mô hình CNN cho text classification"""
    
    model = keras.Sequential([
        # Embedding layer
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            name='embedding'
        ),
        
        # Spatial Dropout để tránh overfitting
        layers.SpatialDropout1D(0.2, name='spatial_dropout')
    ])
    
    # Multiple Convolutional layers với different kernel sizes
    # Branch 1: kernel size 3
    model.add(layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_3'))
    model.add(layers.BatchNormalization(name='batch_norm_3'))
    model.add(layers.MaxPooling1D(pool_size=2, name='maxpool_3'))
    model.add(layers.Dropout(0.3, name='dropout_3'))
    
    # Branch 2: kernel size 5
    model.add(layers.Conv1D(128, 5, activation='relu', padding='same', name='conv1d_5'))
    model.add(layers.BatchNormalization(name='batch_norm_5'))
    model.add(layers.MaxPooling1D(pool_size=2, name='maxpool_5'))
    model.add(layers.Dropout(0.3, name='dropout_5'))
    
    # Branch 3: kernel size 7
    model.add(layers.Conv1D(256, 7, activation='relu', padding='same', name='conv1d_7'))
    model.add(layers.BatchNormalization(name='batch_norm_7'))
    model.add(layers.Dropout(0.3, name='dropout_7'))
    
    # Global Max Pooling
    model.add(layers.GlobalMaxPooling1D(name='global_maxpool'))
    
    # Dense layers
    model.add(layers.Dense(256, activation='relu', name='dense_1'))
    model.add(layers.BatchNormalization(name='batch_norm_dense'))
    model.add(layers.Dropout(0.5, name='dropout_dense'))
    
    model.add(layers.Dense(128, activation='relu', name='dense_2'))
    model.add(layers.Dropout(0.5, name='dropout_dense2'))
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        loss = 'sparse_categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def train_cnn_model(X_train, y_train, vocab_size, max_len, embedding_dim=100, epochs=50, batch_size=32):
    """Train mô hình CNN"""
    print("🏗️ Building CNN model...")
    
    # Xây dựng model
    num_classes = 2  # Binary classification
    
    cnn_model = build_cnn_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )
    
    print(f"✓ CNN model built successfully!")
    print(f"✓ Total parameters: {cnn_model.count_params():,}")
    cnn_model.summary()
    
    # Callbacks
    print("\n🚀 Setting up training callbacks...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_cnn_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured successfully!")
    
    # Class weights
    print("\n⚖️ Computing class weights...")
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
    
    print(f"✓ Class weights: {class_weight_dict}")
    print(f"✓ Class distribution: {np.bincount(y_train)}")
    
    # Training
    print("\n🚀 Bắt đầu training CNN...")
    
    start_time = time.time()
    
    history = cnn_model.fit(
        X_train, y_train,
        validation_split=0.2,  # 20% validation từ training data
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds")
    
    return cnn_model, history, training_time

def evaluate_cnn_model(cnn_model, X_test, y_test, training_time):
    """Đánh giá mô hình CNN"""
    print("📊 Đánh giá mô hình CNN...")
    
    # Prediction
    pred_start = time.time()
    y_cnn_pred_proba = cnn_model.predict(X_test, batch_size=512)
    prediction_time = time.time() - pred_start
    
    # Convert probabilities to predictions
    if y_cnn_pred_proba.shape[1] == 1:  # Binary classification
        y_cnn_pred = (y_cnn_pred_proba > 0.5).astype(int).flatten()
    else:  # Multi-class
        y_cnn_pred = np.argmax(y_cnn_pred_proba, axis=1)
    
    # Calculate metrics
    cnn_accuracy = accuracy_score(y_test, y_cnn_pred)
    cnn_precision = precision_score(y_test, y_cnn_pred, average='weighted')
    cnn_recall = recall_score(y_test, y_cnn_pred, average='weighted')
    cnn_f1 = f1_score(y_test, y_cnn_pred, average='weighted')
    
    # ROC AUC
    try:
        if y_cnn_pred_proba.shape[1] == 1:
            cnn_auc = roc_auc_score(y_test, y_cnn_pred_proba.flatten())
        else:
            cnn_auc = roc_auc_score(y_test, y_cnn_pred_proba, multi_class='ovr')
    except:
        cnn_auc = None
    
    print(f"✓ CNN Performance Metrics:")
    print(f"  • Accuracy: {cnn_accuracy:.4f}")
    print(f"  • Precision: {cnn_precision:.4f}")
    print(f"  • Recall: {cnn_recall:.4f}")
    print(f"  • F1-Score: {cnn_f1:.4f}")
    if cnn_auc:
        print(f"  • ROC AUC: {cnn_auc:.4f}")
    print(f"  • Training time: {training_time:.2f}s")
    print(f"  • Prediction time: {prediction_time:.4f}s")
    
    return y_cnn_pred, y_cnn_pred_proba, {
        'accuracy': cnn_accuracy,
        'precision': cnn_precision,
        'recall': cnn_recall,
        'f1': cnn_f1,
        'auc': cnn_auc,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

def plot_confusion_matrix(y_test, y_pred):
    """Vẽ confusion matrix"""
    print("📊 Confusion Matrix cho CNN:")
    cm_cnn = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Original (0)', 'Computer Generated (1)'], 
                yticklabels=['Original (0)', 'Computer Generated (1)'])
    plt.title('Confusion Matrix - CNN Text Classifier', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Original (0)', 'Computer Generated (1)']))

def plot_training_history(history):
    """Vẽ biểu đồ training history"""
    print("📈 Vẽ biểu đồ training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('CNN Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('CNN Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (nếu có)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
    
    # Training Progress
    epochs = range(1, len(history.history['accuracy']) + 1)
    axes[1, 1].plot(epochs, history.history['accuracy'], 'b-', label='Training')
    if 'val_accuracy' in history.history:
        axes[1, 1].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation')
    axes[1, 1].set_title('CNN Training Progress')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def save_cnn_components(cnn_model, tokenizer, label_map, preprocessing_params):
    """Lưu các thành phần của CNN"""
    print("💾 Lưu mô hình CNN...")
    
    # Lưu model
    cnn_model.save('cnn_text_classifier_final.h5')
    print("✓ CNN model saved to 'cnn_text_classifier_final.h5'")
    
    # Lưu tokenizer
    with open('cnn_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("✓ CNN tokenizer saved to 'cnn_tokenizer.pkl'")
    
    # Lưu label mapping
    with open('cnn_label_mapping.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    print("✓ CNN label mapping saved to 'cnn_label_mapping.pkl'")
    
    # Lưu preprocessing parameters
    with open('cnn_preprocessing_params.pkl', 'wb') as f:
        pickle.dump(preprocessing_params, f)
    print("✓ CNN preprocessing parameters saved to 'cnn_preprocessing_params.pkl'")
    
    print("\n✅ All CNN components saved successfully!")

def main():
    """Main function để chạy toàn bộ pipeline"""
    print("🎯 CNN FAKE REVIEWS DETECTION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # 1. Load và explore data
    df = load_and_explore_data()
    
    # 2. Clean data
    df = clean_data(df)
    
    # 3. Analyze labels
    df = analyze_labels(df)
    
    # 4. Preprocess text cho CNN
    df = preprocess_text_for_cnn(df)
    
    # 5. Analyze text length
    df = analyze_text_length(df)
    
    # 6. Prepare sequences cho CNN
    X_padded, tokenizer, MAX_WORDS, MAX_LEN = prepare_sequences_for_cnn(df)
    
    # 7. Prepare labels
    y, label_map = prepare_labels(df)
    
    # 8. Split data
    X_train, X_test, y_train, y_test = split_data(X_padded, y)
    
    # 9. Train CNN model
    cnn_model, history, training_time = train_cnn_model(
        X_train, y_train, MAX_WORDS, MAX_LEN, 
        embedding_dim=100, epochs=50, batch_size=32
    )
    
    # 10. Evaluate model
    y_pred, y_pred_proba, metrics = evaluate_cnn_model(
        cnn_model, X_test, y_test, training_time
    )
    
    # 11. Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # 12. Plot training history
    plot_training_history(history)
    
    # 13. Save components
    preprocessing_params = {
        'MAX_WORDS': MAX_WORDS,
        'MAX_LEN': MAX_LEN,
        'EMBEDDING_DIM': 100
    }
    save_cnn_components(cnn_model, tokenizer, label_map, preprocessing_params)
    
    # 14. Final summary
    print("\n🎯 CNN FAKE REVIEWS DETECTION - FINAL SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 CNN PERFORMANCE:")
    print(f"• Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"• Precision: {metrics['precision']:.4f}")
    print(f"• Recall: {metrics['recall']:.4f}")
    print(f"• F1-Score: {metrics['f1']:.4f}")
    if metrics['auc']:
        print(f"• ROC AUC: {metrics['auc']:.4f}")
    
    print(f"\n⏱️ TIMING:")
    print(f"• Training time: {metrics['training_time']:.2f} seconds")
    print(f"• Prediction time: {metrics['prediction_time']:.4f} seconds")
    print(f"• Training efficiency: {metrics['accuracy']/metrics['training_time']:.3f}")
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"• Vocabulary size: {MAX_WORDS:,}")
    print(f"• Max sequence length: {MAX_LEN}")
    print(f"• Total parameters: {cnn_model.count_params():,}")
    
    print(f"\n💾 SAVED FILES:")
    print(f"• CNN Model: cnn_text_classifier_final.h5")
    print(f"• Tokenizer: cnn_tokenizer.pkl")
    print(f"• Label Mapping: cnn_label_mapping.pkl")
    print(f"• Preprocessing Params: cnn_preprocessing_params.pkl")
    
    print(f"\n✅ CNN ANALYSIS HOÀN THÀNH!")
    print(f"   Model đã được lưu và sẵn sàng sử dụng! 🚀")

if __name__ == "__main__":
    main()
