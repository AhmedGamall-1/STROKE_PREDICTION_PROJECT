
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import packaging
# Set page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - High Contrast Theme
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* High risk alert box */
    .risk-high {
        background-color: #fff5f5;
        color: #c53030;
        border: 2px solid #fc8181;
        border-left: 6px solid #e53e3e;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(229, 62, 62, 0.2);
    }
    
    /* Low risk success box */
    .risk-low {
        background-color: #f0fff4;
        color: #2f855a;
        border: 2px solid #9ae6b4;
        border-left: 6px solid #38a169;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(56, 161, 105, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Streamlit elements styling */
    .stApp {
        background-color: #f7fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Labels */
    .stSelectbox label, .stSlider label {
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 8px !important;
        color: #2d3748 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3182ce !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #2c5aa0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(49, 130, 206, 0.4) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #edf2f7 !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Text color fixes */
    .css-10trblm {
        color: #2d3748 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #3182ce !important;
    }
    
    /* Input text color */
    .stTextInput > div > div > input {
        color: #2d3748 !important;
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load the trained model and preprocessing components"""
    try:
        # Load the AdaBoost model
        model = joblib.load('adaboost_stroke_prediction_model.pkl')

        # Load the scaler
        scaler = joblib.load('feature_scaler.pkl')

        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        # Load model metadata
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        return model, scaler, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None, None, None, None

def encode_categorical_features(data):
    """Encode categorical features to match training data format"""
    # Create a copy of the data
    encoded_data = data.copy()

    # Encode binary categorical variables
    if 'gender' in encoded_data.columns:
        gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
        encoded_data['gender'] = gender_map.get(encoded_data['gender'].iloc[0], 0)

    if 'ever_married' in encoded_data.columns:
        married_map = {'Yes': 1, 'No': 0}
        encoded_data['ever_married'] = married_map.get(encoded_data['ever_married'].iloc[0], 0)

    if 'Residence_type' in encoded_data.columns:
        residence_map = {'Urban': 1, 'Rural': 0}
        encoded_data['Residence_type'] = residence_map.get(encoded_data['Residence_type'].iloc[0], 0)

    return encoded_data

def create_dummy_features(data, feature_names):
    """Create dummy variables to match the training feature set"""
    # Initialize a DataFrame with all required features set to 0
    full_data = pd.DataFrame(0, index=[0], columns=feature_names)

    # Fill in the values from input data
    for col in data.columns:
        if col in full_data.columns:
            full_data[col] = data[col].iloc[0]

    # Handle work_type dummies
    work_type = data['work_type'].iloc[0] if 'work_type' in data.columns else 'Private'
    work_type_dummies = {
        'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_type_children': 1 if work_type == 'children' else 0
    }
    for col, val in work_type_dummies.items():
        if col in full_data.columns:
            full_data[col] = val

    # Handle smoking_status dummies
    smoking_status = data['smoking_status'].iloc[0] if 'smoking_status' in data.columns else 'never smoked'
    smoking_dummies = {
        'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0
    }
    for col, val in smoking_dummies.items():
        if col in full_data.columns:
            full_data[col] = val

    # Handle age_category dummies
    age = data['age'].iloc[0] if 'age' in data.columns else 45
    if age <= 30:
        age_cat = 'Young'
    elif age <= 45:
        age_cat = 'Adult'
    elif age <= 60:
        age_cat = 'Middle_aged'
    elif age <= 75:
        age_cat = 'Senior'
    else:
        age_cat = 'Elderly'

    age_dummies = {
        'age_category_Adult': 1 if age_cat == 'Adult' else 0,
        'age_category_Middle_aged': 1 if age_cat == 'Middle_aged' else 0,
        'age_category_Senior': 1 if age_cat == 'Senior' else 0,
        'age_category_Elderly': 1 if age_cat == 'Elderly' else 0
    }
    for col, val in age_dummies.items():
        if col in full_data.columns:
            full_data[col] = val

    # Handle BMI category dummies
    bmi = data['bmi'].iloc[0] if 'bmi' in data.columns else 25
    if bmi < 18.5:
        bmi_cat = 'Underweight'
    elif bmi < 25:
        bmi_cat = 'Normal'
    elif bmi < 30:
        bmi_cat = 'Overweight'
    else:
        bmi_cat = 'Obese'

    bmi_dummies = {
        'bmi_category_Obese': 1 if bmi_cat == 'Obese' else 0,
        'bmi_category_Overweight': 1 if bmi_cat == 'Overweight' else 0,
        'bmi_category_Underweight': 1 if bmi_cat == 'Underweight' else 0
    }
    for col, val in bmi_dummies.items():
        if col in full_data.columns:
            full_data[col] = val

    # Handle glucose category dummies
    glucose = data['avg_glucose_level'].iloc[0] if 'avg_glucose_level' in data.columns else 90
    if glucose < 100:
        glucose_cat = 'Normal'
    elif glucose < 126:
        glucose_cat = 'Prediabetes'
    else:
        glucose_cat = 'Diabetes'

    glucose_dummies = {
        'glucose_category_Normal': 1 if glucose_cat == 'Normal' else 0,
        'glucose_category_Prediabetes': 1 if glucose_cat == 'Prediabetes' else 0
    }
    for col, val in glucose_dummies.items():
        if col in full_data.columns:
            full_data[col] = val

    return full_data

def calculate_advanced_features(data):
    """Calculate advanced engineered features"""
    # Get basic values
    age = data['age'].iloc[0]
    hypertension = data['hypertension'].iloc[0]
    heart_disease = data['heart_disease'].iloc[0]
    avg_glucose_level = data['avg_glucose_level'].iloc[0]
    bmi = data['bmi'].iloc[0]
    gender = data['gender'].iloc[0]

    # Calculate advanced features
    data['risk_score'] = (
        age * 0.1 +
        hypertension * 10 +
        heart_disease * 15 +
        avg_glucose_level * 0.05 +
        bmi * 0.5
    )

    # Medical Risk Interaction Features
    data['hypertension_heart_disease'] = hypertension * heart_disease
    data['age_hypertension_interaction'] = age * hypertension
    data['diabetes_hypertension'] = (1 if avg_glucose_level > 126 else 0) * hypertension
    data['triple_cardiovascular_risk'] = hypertension * heart_disease * (1 if age > 65 else 0)

    # Age-Based Risk Stratification
    data['age_risk_score'] = 3 if age > 75 else (2 if age > 65 else (1 if age > 55 else 0))
    data['elderly_with_conditions'] = (1 if age > 70 else 0) * (hypertension + heart_disease)
    data['young_with_high_glucose'] = (1 if age < 45 else 0) * (1 if avg_glucose_level > 140 else 0)

    # BMI-Glucose Metabolic Features
    data['metabolic_syndrome_score'] = (1 if bmi > 30 else 0) + (1 if avg_glucose_level > 100 else 0) + hypertension
    data['obesity_diabetes'] = (1 if bmi > 30 else 0) * (1 if avg_glucose_level > 126 else 0)
    data['bmi_glucose_ratio'] = bmi / (avg_glucose_level / 100)

    # Gender-Specific Risk Features
    data['male_early_risk'] = (1 if gender == 1 else 0) * (1 if age > 45 else 0)
    data['female_post_menopause'] = (1 if gender == 0 else 0) * (1 if age > 50 else 0)

    # Advanced Mathematical Features
    data['age_squared'] = age ** 2
    data['glucose_log'] = np.log(avg_glucose_level + 1)
    data['bmi_age_product'] = bmi * age

    # Medical Threshold Features
    data['critical_glucose'] = 1 if avg_glucose_level > 200 else 0
    data['morbid_obesity'] = 1 if bmi > 40 else 0
    data['geriatric_patient'] = 1 if age > 80 else 0
    data['multiple_conditions'] = 1 if (hypertension + heart_disease + (1 if bmi > 30 else 0) >= 2) else 0

    # Composite Health Scores
    data['framingham_risk_proxy'] = (
        (age - 30) * 0.1 + 
        hypertension * 15 + 
        (avg_glucose_level - 100) * 0.1 + 
        (bmi - 25) * 0.5
    )

    data['stroke_risk_index'] = (
        age * 0.15 +
        hypertension * 20 +
        heart_disease * 25 +
        (1 if avg_glucose_level > 126 else 0) * 15 +
        (1 if bmi > 30 else 0) * 10
    )

    return data

def main():
    # Load model components
    model, scaler, feature_names, metadata = load_model_components()

    if model is None:
        st.error("❌ Failed to load model components. Please ensure all model files are present.")
        return

    # Main header
    st.markdown('<h1 class="main-header">🏥 Stroke Risk Prediction System</h1>', unsafe_allow_html=True)

    # Model information sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        if metadata:
            st.info(f"""
            **Model**: {metadata['model_name']}
            **Accuracy**: {metadata['performance_metrics']['accuracy']:.3f}
            **F1-Score**: {metadata['performance_metrics']['f1_score']:.3f}
            **ROC-AUC**: {metadata['performance_metrics']['roc_auc']:.3f}
            **Training Date**: {metadata['training_date']}
            """)

    # Create input form
    st.header("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 0, 100, 45)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    with col2:
        st.subheader("Medical Information")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0, 0.1)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Prediction button
    if st.button("🔍 Predict Stroke Risk", type="primary"):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })

            # Calculate advanced features
            input_data = calculate_advanced_features(input_data)

            # Encode categorical features
            input_encoded = encode_categorical_features(input_data)

            # Create full feature set with dummy variables
            input_full = create_dummy_features(input_encoded, feature_names)

            # Scale features
            input_scaled = scaler.transform(input_full)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Display results
            st.header("🎯 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Stroke Risk", "High Risk" if prediction == 1 else "Low Risk", 
                         delta=f"{prediction_proba[1]*100:.1f}% probability")

            with col2:
                st.metric("No Stroke Probability", f"{prediction_proba[0]*100:.1f}%")

            with col3:
                st.metric("Stroke Probability", f"{prediction_proba[1]*100:.1f}%")

            # Risk assessment
            risk_level = prediction_proba[1]

            if risk_level > 0.5:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ HIGH RISK ALERT</h3>
                    <p>The model indicates a <strong>{risk_level*100:.1f}%</strong> probability of stroke risk.</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                        <li>Consult with a healthcare provider immediately</li>
                        <li>Consider comprehensive cardiovascular evaluation</li>
                        <li>Monitor blood pressure and glucose levels regularly</li>
                        <li>Implement lifestyle modifications</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ LOW RISK</h3>
                    <p>The model indicates a <strong>{risk_level*100:.1f}%</strong> probability of stroke risk.</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                        <li>Continue healthy lifestyle practices</li>
                        <li>Regular health check-ups</li>
                        <li>Monitor risk factors periodically</li>
                        <li>Maintain healthy diet and exercise routine</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(x=['No Stroke', 'Stroke'], 
                      y=[prediction_proba[0]*100, prediction_proba[1]*100],
                      marker_color=['green', 'red'])
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk factors analysis
            st.header("📈 Risk Factors Analysis")

            risk_factors = []
            if age > 65:
                risk_factors.append(f"Advanced age ({age} years)")
            if hypertension:
                risk_factors.append("Hypertension")
            if heart_disease:
                risk_factors.append("Heart disease")
            if avg_glucose_level > 126:
                risk_factors.append(f"High glucose level ({avg_glucose_level:.1f} mg/dL)")
            if bmi > 30:
                risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
            if smoking_status == "smokes":
                risk_factors.append("Current smoker")

            if risk_factors:
                st.warning("⚠️ **Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ **No major risk factors identified**")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
