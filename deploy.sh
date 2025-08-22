#!/bin/bash
# Streamlit Stroke Prediction App Deployment Script

echo "🚀 Setting up Stroke Prediction App..."

# Install requirements
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Check if model files exist
echo "🔍 Checking model files..."
if [ ! -f "adaboost_stroke_prediction_model.pkl" ]; then
    echo "❌ Error: adaboost_stroke_prediction_model.pkl not found!"
    echo "Please run the Jupyter notebook first to train and save the model."
    exit 1
fi

if [ ! -f "feature_scaler.pkl" ]; then
    echo "❌ Error: feature_scaler.pkl not found!"
    echo "Please run the Jupyter notebook first to save the scaler."
    exit 1
fi

if [ ! -f "feature_names.pkl" ]; then
    echo "❌ Error: feature_names.pkl not found!"
    echo "Please run the Jupyter notebook first to save feature names."
    exit 1
fi

if [ ! -f "model_metadata.pkl" ]; then
    echo "❌ Error: model_metadata.pkl not found!"
    echo "Please run the Jupyter notebook first to save model metadata."
    exit 1
fi

echo "✅ All model files found!"

# Launch Streamlit app
echo "🌐 Launching Streamlit app..."
streamlit run streamlit_stroke_app.py
