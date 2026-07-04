
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "harshverma27/predictive-maintenance-engine"

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pkl")
    scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename="scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

st.title("Engine Health Predictor")
st.write("Enter live sensor readings to predict engine condition.")

rpm = st.number_input("Engine rpm", min_value=0.0, value=750.0)
oil_p = st.number_input("Lub oil pressure", min_value=0.0, value=3.0)
fuel_p = st.number_input("Fuel pressure", min_value=0.0, value=6.5)
coolant_p = st.number_input("Coolant pressure", min_value=0.0, value=2.3)
oil_t = st.number_input("lub oil temp", min_value=0.0, value=77.0)
coolant_t = st.number_input("Coolant temp", min_value=0.0, value=78.0)

if st.button("Predict Engine Condition"):
    input_df = pd.DataFrame([{
        "Engine rpm": rpm,
        "Lub oil pressure": oil_p,
        "Fuel pressure": fuel_p,
        "Coolant pressure": coolant_p,
        "lub oil temp": oil_t,
        "Coolant temp": coolant_t
    }])
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if pred == 1:
        st.error(f"Faulty - maintenance required (fault probability: {prob:.2%})")
    else:
        st.success(f"Normal operation (normal confidence: {(1 - prob):.2%})")
