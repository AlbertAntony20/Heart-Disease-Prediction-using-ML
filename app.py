import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient clinical details")

# ---------------- INPUT FIELDS ----------------

age = st.number_input("Age (years)", 1, 120, 45)

sex_label = st.selectbox(
    "Sex",
    ["Female", "Male"]
)
sex = 0 if sex_label == "Female" else 1

cp_label = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_map[cp_label]

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    100, 600, 200
)

fbs_label = st.selectbox(
    "Fasting Blood Sugar",
    ["≤ 120 mg/dl", "> 120 mg/dl"]
)
fbs = 1 if fbs_label == "> 120 mg/dl" else 0

restecg_label = st.selectbox(
    "Resting ECG Results",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_map[restecg_label]

thalch = st.number_input(
    "Maximum Heart Rate Achieved",
    60, 220, 150
)

exang_label = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.number_input(
    "ST Depression Induced by Exercise",
    0.0, 6.0, 1.0
)

slope_label = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope = slope_map[slope_label]

ca_label = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    ["0", "1", "2", "3"]
)
ca = int(ca_label)

thal_label = st.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"]
)
thal_map = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Unknown": 3
}
thal = thal_map[thal_label]

# ---------------- PREDICTION ----------------

if st.button("Predict Heart Disease"):
    input_data = np.array([
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalch, exang, oldpeak,
        slope, ca, thal
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] > 0:
        st.error("⚠️ High Risk: Heart Disease Detected")
    else:
        st.success("✅ Low Risk: No Heart Disease Detected")
