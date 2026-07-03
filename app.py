"""
Heart Disease Risk Predictor
A Streamlit app that predicts heart disease risk from 14 clinical features,
explains the prediction with SHAP, and generates a plain-language explanation
using Google's Gemini 2.5 Flash (free tier).
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = "model/heart_model.pkl"

# ---------------------------------------------------------------------------
# Load model artifact (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

artifact = load_artifact()

# ---------------------------------------------------------------------------
# Gemini setup (API key comes from Streamlit secrets, never hardcoded)
# ---------------------------------------------------------------------------
def get_gemini_model():
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


def get_llm_explanation(top_factors: dict, prediction_label: str, probability: float):
    """Calls Gemini to turn SHAP values into a plain-language explanation.
    Falls back gracefully if the API key is missing or the call fails/rate-limits."""
    model = get_gemini_model()
    if model is None:
        return None

    factors_text = ", ".join(
        f"{name} ({'increases' if val > 0 else 'decreases'} risk)"
        for name, val in top_factors.items()
    )

    prompt = f"""
    A machine learning model analyzed a patient's clinical data and predicted: {prediction_label}
    (estimated probability: {probability:.0%}).

    The top contributing factors identified by the model were: {factors_text}

    Write a short, plain-language explanation (3-4 sentences) of WHY the model
    likely reached this result, based only on these factors. Do not add any
    medical advice, treatment suggestions, or diagnosis beyond explaining the
    model's own reasoning. Keep it neutral and easy to understand for a
    non-technical reader.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None  # caller falls back to the raw SHAP breakdown


# ---------------------------------------------------------------------------
# Feature metadata: label, help text, and input widget config
# Matches the standard UCI Cleveland Heart Disease encoding.
# ---------------------------------------------------------------------------
FEATURE_INFO = {
    "age": {"label": "Age", "help": "Age in years.", "min": 1, "max": 120, "default": 50},
    "sex": {"label": "Sex", "help": "Biological sex.", "options": {"Male": 1, "Female": 0}},
    "cp": {
        "label": "Chest Pain Type",
        "help": "Type of chest pain experienced.",
        "options": {
            "Typical angina": 1,
            "Atypical angina": 2,
            "Non-anginal pain": 3,
            "Asymptomatic": 4,
        },
    },
    "trestbps": {
        "label": "Resting Blood Pressure (mm Hg)",
        "help": "Blood pressure measured at rest, on admission to hospital.",
        "min": 60, "max": 250, "default": 120,
    },
    "chol": {
        "label": "Serum Cholesterol (mg/dl)",
        "help": "Cholesterol level from a blood test.",
        "min": 100, "max": 600, "default": 200,
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl?",
        "help": "Whether fasting blood sugar exceeds 120 mg/dl.",
        "options": {"No": 0, "Yes": 1},
    },
    "restecg": {
        "label": "Resting ECG Results",
        "help": "Resting electrocardiographic results.",
        "options": {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2,
        },
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "help": "Highest heart rate reached during a stress test.",
        "min": 60, "max": 220, "default": 150,
    },
    "exang": {
        "label": "Exercise-Induced Angina?",
        "help": "Chest pain triggered specifically by exercise.",
        "options": {"No": 0, "Yes": 1},
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "help": "ST depression induced by exercise, relative to rest.",
        "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1,
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "help": "Shape of the ST segment during peak exercise.",
        "options": {"Upsloping": 1, "Flat": 2, "Downsloping": 3},
    },
    "ca": {
        "label": "Major Vessels Colored by Fluoroscopy",
        "help": "Number of major blood vessels (0-3) visible via fluoroscopy.",
        "options": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "thal": {
        "label": "Thalassemia",
        "help": "Blood disorder test result.",
        "options": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7},
    },
}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def humanize_feature_name(encoded_name: str) -> str:
    """Converts an encoded feature name like 'ca_0.0' or 'thal_7.0' into a
    readable label like '0 (Major Vessels Colored by Fluoroscopy)' or
    'Reversible defect (Thalassemia)'. Numeric features pass through using
    their display label."""
    if encoded_name in FEATURE_INFO:
        return FEATURE_INFO[encoded_name]["label"]

    for key, info in FEATURE_INFO.items():
        prefix = f"{key}_"
        if encoded_name.startswith(prefix) and "options" in info:
            raw_value = encoded_name[len(prefix):]
            try:
                raw_value_num = float(raw_value)
            except ValueError:
                continue
            for option_label, option_value in info["options"].items():
                if float(option_value) == raw_value_num:
                    return f"{option_label} ({info['label']})"

    return encoded_name  # fallback, shouldn't normally happen

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")

if artifact is None:
    st.error(
        "Model file not found. Run `python train_model.py` locally first to "
        "generate `model/heart_model.pkl`, then commit it (or let your deploy "
        "step train it automatically — see README)."
    )
    st.stop()

st.subheader("Enter Patient Information")

with st.form("patient_form"):
    inputs = {}
    col1, col2 = st.columns(2)
    columns = [col1, col2]

    for i, (key, info) in enumerate(FEATURE_INFO.items()):
        target_col = columns[i % 2]
        with target_col:
            if "options" in info:
                choice = st.selectbox(
                    info["label"], list(info["options"].keys()), help=info["help"], key=key
                )
                inputs[key] = info["options"][choice]
            else:
                step = info.get("step", 1)
                value_type = float if isinstance(step, float) else int
                inputs[key] = st.number_input(
                    info["label"],
                    min_value=value_type(info["min"]),
                    max_value=value_type(info["max"]),
                    value=value_type(info["default"]),
                    step=step,
                    help=info["help"],
                    key=key,
                )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Validation (defense in depth beyond the widget constraints themselves)
# ---------------------------------------------------------------------------
def validate_inputs(data: dict) -> list:
    errors = []
    for key, info in FEATURE_INFO.items():
        if key not in data or data[key] is None:
            errors.append(f"Missing value for '{info['label']}'.")
            continue
        if "min" in info:
            if not (info["min"] <= data[key] <= info["max"]):
                errors.append(
                    f"'{info['label']}' must be between {info['min']} and {info['max']}."
                )
        if "options" in info:
            if data[key] not in info["options"].values():
                errors.append(f"'{info['label']}' has an invalid selection.")
    return errors


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------
if submitted:
    errors = validate_inputs(inputs)

    if errors:
        st.error("Please fix the following before continuing:")
        for e in errors:
            st.write(f"- {e}")
        st.stop()

    with st.spinner("Analyzing..."):
        X = pd.DataFrame([inputs])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        try:
            X_proc = artifact["preprocessor"].transform(X)
        except Exception as e:
            st.error(f"Could not process this input — please check your values. ({e})")
            st.stop()

        # Prediction (calibrated probability)
        proba = artifact["calibrated_model"].predict_proba(X_proc)[0, 1]
        prediction_label = "Higher Risk of Heart Disease" if proba >= 0.5 else "Lower Risk of Heart Disease"

        # Anomaly check
        anomaly_score = artifact["anomaly_detector"].predict(X_proc)[0]
        is_anomaly = anomaly_score == -1

        # SHAP explanation — pick the right explainer type for the model family
        try:
            model_name = artifact.get("model_name", "")
            shap_model = artifact["shap_model"]
            feature_names = artifact["encoded_feature_names"]

            if model_name == "LogisticRegression":
                explainer = shap.LinearExplainer(shap_model, artifact["background_sample"])
                raw_values = explainer.shap_values(X_proc)
                values = np.array(raw_values)[0]
            else:  # RandomForest, XGBoost -> tree-based
                explainer = shap.TreeExplainer(shap_model)
                raw_values = explainer.shap_values(X_proc)
                if isinstance(raw_values, list):
                    values = np.array(raw_values[-1])[0]
                else:
                    arr = np.array(raw_values)
                    values = arr[0][:, -1] if arr.ndim == 3 else arr[0]

            shap_pairs = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)
            top_factors = {
                humanize_feature_name(name): float(val) for name, val in shap_pairs[:5]
            }
        except Exception:
            top_factors = {}

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Result")

    risk_color = "🔴" if proba >= 0.5 else "🟢"
    st.markdown(f"### {risk_color} {prediction_label}")
    st.progress(min(max(proba, 0.0), 1.0), text=f"Estimated probability: {proba:.0%}")

    if is_anomaly:
        st.warning(
            "⚠️ This combination of values is unusual compared to the data the model "
            "was trained on — treat this prediction with extra caution."
        )

    if top_factors:
        st.subheader("Top Contributing Factors")
        fig, ax = plt.subplots(figsize=(6, 3))
        names = list(top_factors.keys())
        vals = list(top_factors.values())
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("Impact on prediction (SHAP value)")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.subheader("AI-Generated Explanation")
        llm_text = get_llm_explanation(top_factors, prediction_label, proba)
        if llm_text:
            st.info(llm_text)
        else:
            st.caption(
                "AI explanation unavailable right now (missing API key or rate limit reached) "
                "— showing the raw factor breakdown above instead."
            )

    st.divider()
    st.caption(
        "⚕️ This tool is for educational purposes only and is not a substitute for "
        "professional medical diagnosis. Please consult a doctor for real health concerns."
    )
