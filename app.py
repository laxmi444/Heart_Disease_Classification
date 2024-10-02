import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"C:\Users\Omkar\Heart_Disease_Classification\heart_disease_classifier.pkl")


# Define the Streamlit app
def main():
    st.title("Heart Disease Prediction 🫀")

    st.write("""
    This app predicts whether a patient is likely to have heart disease based on their input parameters.
    """)

    # User inputs for model prediction
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", (1, 0))
    
    cp = st.selectbox("Chest Pain Type", 
                      options=[0, 1, 2, 3], 
                      format_func=lambda x: {0: "Typical Angina", 
                                             1: "Atypical Angina", 
                                             2: "Non-anginal Pain", 
                                             3: "Asymptomatic"}[x])
    
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", (1, 0))
    
    restecg = st.selectbox("Resting ECG", 
                           options=[0, 1, 2], 
                           format_func=lambda x: {0: "Normal", 
                                                  1: "ST-T Wave Abnormality", 
                                                  2: "Left Ventricular Hypertrophy"}[x])
    
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", (1, 0))
    
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", 
                         options=[0, 1, 2], 
                         format_func=lambda x: {0: "Upsloping", 
                                                1: "Flat", 
                                                2: "Downsloping"}[x])
    
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    
    thal = st.selectbox("Thalassemia", 
                        options=[1, 3, 6, 7], 
                        format_func=lambda x: {1: "Normal", 
                                               3: "Normal", 
                                               6: "Fixed Defect", 
                                               7: "Reversible Defect"}[x])
    
    # Prepare input data for prediction
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # Predict button
    if st.button("Predict"):
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.success("The patient is likely to have heart disease.")
        else:
            st.success("The patient is unlikely to have heart disease.")

if __name__ == '__main__':
    main()
