# ❤️ Heart Disease Risk Predictor

### 🚀 Live Demo
🔗 **https://heartdiseaseprediction-clc5f5pehylyxlxquvsi44.streamlit.app/**

An educational Machine Learning web application that predicts heart disease risk using 14 clinical features, explains predictions with **SHAP**, and generates an easy-to-understand explanation using **Google Gemini 2.5 Flash**.

> ⚕️ **Disclaimer:** This project is for educational purposes only. It is **not** a medical diagnostic tool and should **not** be used for clinical decision-making. Always consult a qualified healthcare professional for medical advice.

---

# 📑 Table of Contents

- 🚀 Live Demo
- ✨ Features
- 📂 Project Structure
- 📊 Input Features
- ✅ Input Validation & Error Handling
- 🤖 Model Training Pipeline
- 💻 Running Locally
- ☁️ Deploying to Streamlit Community Cloud
- 📦 Pushing to GitHub
- 🛠️ Tech Stack

---

# ✨ Features

- ❤️ Predicts heart disease risk from 13 clinical inputs
- 📈 Displays prediction probability
- 🔍 Explains predictions using SHAP values
- 🤖 Generates plain-language explanations using Google Gemini 2.5 Flash
- ⚠️ Detects unusual inputs using Isolation Forest
- ✅ Robust input validation
- ☁️ One-click deployment on Streamlit Cloud

---

# 📂 Project Structure

```text
heart-disease-predictor/
├── data/
│   └── heart.csv
├── model/
│   └── heart_model.pkl
├── .streamlit/
│   └── secrets.toml.example
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
