import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# ---------- load pipeline (preprocessing + model) ----------
try:
    model = joblib.load("models/final_model.pkl")
except Exception as e:
    st.error("Cannot load model: " + str(e))
    st.stop()

st.write("Enter patient data below (use realistic values).")

# ---------- Inputs: make sure we collect ALL features the pipeline expects ----------
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ("Male", "Female"))  # Male=1, Female=0
sex_val = 1 if sex == "Male" else 0

cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4])  # typical uci values 1-4
trestbps = st.number_input("Resting blood pressure (trestbps)", 50, 250, 120)
chol = st.number_input("Serum cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", ("No", "Yes"))
fbs_val = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max heart rate achieved (thalach)", 50, 220, 150)
exang = st.selectbox("Exercise induced angina (exang)", ("No", "Yes"))
exang_val = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the ST segment (slope)", [1, 2, 3])
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca)", 0, 4, 0)
thal = st.selectbox("Thalassemia (thal) (3 = normal, 6 = fixed defect, 7 = reversible defect)", [3, 6, 7])

# ---------- Build DataFrame with EXACT column names/order used in training ----------
input_df = pd.DataFrame([[
    age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach,
    exang_val, oldpeak, slope, ca, thal
]], columns=[
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
])

st.write("Input data preview:")
st.write(input_df)

# ---------- Predict ----------
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            # if binary classification, proba[1] is probability for class 1
            if len(proba) == 2:
                proba = proba[1]
        # Show result
        if pred == 1:
            if proba is not None:
                st.error(f"⚠️ High risk of heart disease (probability = {proba:.2f})")
            else:
                st.error("⚠️ High risk of heart disease")
        else:
            if proba is not None:
                st.success(f"✅ Low risk of heart disease (probability = {proba:.2f})")
            else:
                st.success("✅ Low risk of heart disease")
    except Exception as e:
        st.error("Prediction error: " + str(e))
        # optional debug info
        if st.checkbox("Show debug info"):
            try:
                # show feature names the pipeline expects (if available)
                if hasattr(model, "feature_names_in_"):
                    st.write("model.feature_names_in_:")
                    st.write(model.feature_names_in_)
                pre = model.named_steps.get("preprocessor", None) if hasattr(model, "named_steps") else None
                st.write("preprocessor object:")
                st.write(pre)
                # try to show get_feature_names_out if available
                try:
                    names = pre.get_feature_names_out()
                    st.write("preprocessor.get_feature_names_out():")
                    st.write(names)
                except Exception:
                    st.write("preprocessor.get_feature_names_out() not available in this sklearn version.")
            except Exception as e2:
                st.write("Debug failed: " + str(e2))
