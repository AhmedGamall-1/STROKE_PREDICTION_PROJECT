# 🏥 Stroke Risk Prediction System

A comprehensive machine learning application for predicting stroke risk using patient data, built with Streamlit and powered by an AdaBoost classifier.

## 🎯 Features

- **Interactive Web Interface**: User-friendly form for patient data input
- **Real-time Predictions**: Instant stroke risk assessment with confidence scores
- **Advanced Analytics**: 20+ engineered features for accurate predictions
- **Visual Insights**: Interactive charts and probability visualizations
- **Clinical Decision Support**: Risk factor analysis and medical recommendations
- **Professional Design**: Medical-grade styling with responsive layout

## 📊 Model Performance

- **Algorithm**: AdaBoost Classifier
- **Accuracy**: 90.22%
- **F1-Score**: 0.2647
- **ROC-AUC**: 0.8161

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Required model files (generated from the Jupyter notebook)

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present:**
   - `adaboost_stroke_prediction_model.pkl`
   - `feature_scaler.pkl`
   - `feature_names.pkl`
   - `model_metadata.pkl`

4. **Run the application:**
   ```bash
   streamlit run streamlit_stroke_app.py
   ```

### Alternative: Use the deployment script
```bash
chmod +x deploy.sh
./deploy.sh
```

## 🏥 How to Use

1. **Open the web application** in your browser (usually at http://localhost:8501)
2. **Fill in patient information:**
   - Basic demographics (age, gender, marital status, etc.)
   - Medical history (hypertension, heart disease)
   - Vital signs (glucose level, BMI)
   - Lifestyle factors (smoking status, work type)
3. **Click "Predict Stroke Risk"**
4. **Review the results:**
   - Risk classification (High/Low)
   - Probability scores
   - Risk factor analysis
   - Clinical recommendations

## 📋 Input Parameters

### Demographics
- **Age**: 0-100 years
- **Gender**: Male, Female, Other
- **Marital Status**: Yes/No
- **Residence**: Urban/Rural
- **Work Type**: Private, Self-employed, Government, Children, Never worked

### Medical Information
- **Hypertension**: Yes/No
- **Heart Disease**: Yes/No
- **Average Glucose Level**: 50-300 mg/dL
- **BMI**: 10-50 kg/m²
- **Smoking Status**: Never smoked, Formerly smoked, Currently smokes, Unknown

## 🔬 Technical Details

### Feature Engineering
The application automatically calculates 20+ advanced features including:
- Medical risk interactions (hypertension + heart disease)
- Age-based risk stratification
- Metabolic syndrome indicators
- Gender-specific risk factors
- Mathematical transformations (age², log glucose, etc.)
- Clinical decision thresholds

### Model Architecture
- **Base Algorithm**: AdaBoost (Adaptive Boosting)
- **Feature Count**: 46 engineered features
- **Training Data**: 5,110 patient records
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Preprocessing**: StandardScaler normalization

## ⚠️ Important Disclaimers

- **Educational Purpose**: This tool is designed for educational and research purposes only
- **Not a Medical Device**: This application is not FDA-approved and should not be used for actual medical diagnosis
- **Professional Consultation Required**: Always consult with qualified healthcare professionals for medical decisions
- **Data Privacy**: Ensure patient data privacy and compliance with healthcare regulations (HIPAA, GDPR, etc.)

## 🌐 Deployment Options

### Local Development
```bash
streamlit run streamlit_stroke_app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### Heroku Deployment
1. Add `setup.sh` and `Procfile` for Heroku
2. Deploy using Heroku CLI or GitHub integration

### Docker Deployment
Create a Dockerfile for containerized deployment

## 📈 Model Training

The model was trained using:
- **Dataset**: Healthcare Dataset Stroke Data (5,110 records)
- **Algorithm Comparison**: 8 different ML algorithms tested
- **Cross-validation**: Stratified K-fold validation
- **Hyperparameter Tuning**: Grid search optimization
- **Class Imbalance Handling**: SMOTE oversampling

## 🔧 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure you've run the complete Jupyter notebook to generate model files
   - Check that all .pkl files are in the same directory as the Streamlit app

2. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Performance issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Restart the application

## 📞 Support

For technical support or questions:
- Check the Jupyter notebook for model training details
- Review the Streamlit documentation for deployment issues
- Ensure all dependencies are properly installed

## 🏆 Credits

Developed as part of the ApplAi Stroke Prediction Project
- Machine Learning Model: AdaBoost Classifier
- Web Framework: Streamlit
- Visualization: Plotly
- Data Processing: Pandas, NumPy, Scikit-learn
