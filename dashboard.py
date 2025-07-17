import streamlit as st
import pandas as pd
import joblib
import datetime
import os

# Load the model
# model = joblib.load("model.pkl")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
model = joblib.load(model_path)


st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")
st.title("🚨 Insurance Fraud Detection Dashboard")
st.subheader("🔍 Enter Input Data")

col1, col2 = st.columns(2)

# Numerical Inputs
with col1:
    months_as_customer = float(st.text_input("Months as Customer (0–500)", "120"))
    age = float(st.text_input("Age (18–100)", "45"))
    policy_deductable = float(st.text_input("Policy Deductible (0–5000)", "1000"))
    policy_annual_premium = float(st.text_input("Annual Premium (0–10000)", "850.5"))
    umbrella_limit = float(st.text_input("Umbrella Limit (±1000000)", "0"))
    insured_zip = float(st.text_input("Zip Code (10000–99999)", "43110"))
    capital_gains = float(st.text_input("Capital Gains (0–100000)", "0"))
    capital_loss = float(st.text_input("Capital Loss (0–100000)", "0"))
    incident_hour_of_the_day = float(st.text_input("Incident Hour (0–23)", "14"))
    auto_year = float(st.text_input("Auto Year (1995–2025)", "2015"))

with col2:
    number_of_vehicles_involved = st.selectbox("Vehicles Involved", [1, 2, 3, 4])
    bodily_injuries = st.selectbox("Bodily Injuries", [0, 1, 2])
    witnesses = st.selectbox("Witnesses", [0, 1, 2, 3])
    injury_claim = float(st.text_input("Injury Claim Amount (0–100000)", "5000"))
    property_claim = float(st.text_input("Property Claim Amount (0–100000)", "3000"))
    vehicle_claim = float(st.text_input("Vehicle Claim Amount (0–100000)", "4000"))
    total_claim_amount = float(st.text_input("Total Claim Amount (0–500000)", "12000"))

# Categorical Inputs
cat1, cat2 = st.columns(2)
with cat1:
    policy_state = st.selectbox("Policy State", ["IN", "IL", "OH"])
    policy_csl = st.selectbox("Policy CSL", ["250/500", "500/1000", "100/300"])
    insured_sex = st.selectbox("Sex", ["MALE", "FEMALE"])
    insured_education_level = st.selectbox("Education", ["College", "High School", "JD", "MD", "PhD"])
    insured_occupation = st.selectbox("Occupation", ["engineer", "teacher", "doctor", "clerk", "manager"])
    insured_hobbies = st.selectbox("Hobbies", ["chess", "cross-fit", "reading", "writing", "video games"])
    insured_relationship = st.selectbox("Relationship", ["own-child", "husband", "wife", "other-relative"])
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car"])

with cat2:
    collision_type = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision"])
    incident_severity = st.selectbox("Severity", ["Minor Damage", "Major Damage", "Total Loss"])
    authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Other", "None"])
    incident_state = st.selectbox("Incident State", ["IN", "OH", "IL"])
    incident_city = st.selectbox("Incident City", ["Indianapolis", "Columbus", "Springfield"])
    property_damage = st.selectbox("Property Damage", ["YES", "NO"])
    police_report_available = st.selectbox("Police Report", ["YES", "NO"])
    auto_make = st.selectbox("Auto Make", ["Toyota", "Ford", "Dodge", "Chevrolet"])
    auto_model = st.selectbox("Auto Model", ["Camry", "Escape", "Accord", "Civic", "A3"])

# Prediction Logic
if st.button("🧠 Predict Fraud"):
    input_data = pd.DataFrame([{
        "months_as_customer": months_as_customer,
        "age": age,
        "policy_state": policy_state,
        "policy_csl": policy_csl,
        "policy_deductable": policy_deductable,
        "policy_annual_premium": policy_annual_premium,
        "umbrella_limit": umbrella_limit,
        "insured_zip": insured_zip,
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "insured_occupation": insured_occupation,
        "insured_hobbies": insured_hobbies,
        "insured_relationship": insured_relationship,
        "capital-gains": capital_gains,
        "capital-loss": capital_loss,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": incident_state,
        "incident_city": incident_city,
        "incident_hour_of_the_day": incident_hour_of_the_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "property_damage": property_damage,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": police_report_available,
        "auto_make": auto_make,
        "auto_model": auto_model,
        "auto_year": auto_year,
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "vehicle_claim": vehicle_claim,
        "total_claim_amount": total_claim_amount
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success(f"Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'}")
    st.info(f"Fraud Probability: {round(probability * 100, 2)}%")

    # Logging section
    input_data["prediction"] = prediction
    input_data["fraud_proba"] = probability
    input_data["timestamp"] = datetime.datetime.now()

    os.makedirs("data", exist_ok=True)
    file_path = "data/inference_logs.csv"

    if os.path.exists(file_path):
        logs = pd.read_csv(file_path)
        if not input_data.drop(columns=["timestamp"]).equals(logs.tail(1).drop(columns=["timestamp"])):
            input_data.to_csv(file_path, mode='a', header=False, index=False)
            st.success("✅ New prediction logged.")
        else:
            st.info("ℹ️ Same prediction already exists. Skipped logging.")
    else:
        input_data.to_csv(file_path, mode='a', header=True, index=False)
        st.success("✅ First prediction logged.")

# Option to show logs
if st.checkbox("📄 Show Inference Logs"):
    try:
        df = pd.read_csv("data/inference_logs.csv")
        st.dataframe(df.tail(20))
    except FileNotFoundError:
        st.warning("No logs found yet.")
