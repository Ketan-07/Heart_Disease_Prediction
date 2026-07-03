# ❤️ Heart Disease Risk Predictor

An educational ML app that predicts heart disease risk from 14 clinical features,
explains predictions with SHAP, and generates a plain-language explanation using
Google's **Gemini 2.5 Flash** (free tier).

> ⚕️ **Disclaimer**: This is an educational demo, not a diagnostic tool. It is not
> validated for clinical use. Always consult a qualified doctor for real health concerns.

---

## What's in this project

```
heart-disease-predictor/
├── data/
│   └── heart.csv                    # UCI Cleveland Heart Disease dataset (303 rows, 14 features)
├── model/
│   └── heart_model.pkl              # trained pipeline (created by train_model.py)
├── .streamlit/
│   └── secrets.toml.example         # template — copy to secrets.toml locally, never commit the real one
├── train_model.py                   # trains, compares, calibrates, and saves the model
├── app.py                           # Streamlit app (UI + prediction + SHAP + Gemini explanation)
├── requirements.txt                 # exact pinned versions (matches the versions used to train the model)
├── .gitignore
└── README.md
```

## The 14 input features

| # | Feature | What it means |
|---|---|---|
| 1 | Age | Age in years |
| 2 | Sex | Male / Female |
| 3 | Chest Pain Type | Typical angina / Atypical angina / Non-anginal pain / Asymptomatic |
| 4 | Resting Blood Pressure | mm Hg, measured at rest |
| 5 | Serum Cholesterol | mg/dl, from a blood test |
| 6 | Fasting Blood Sugar > 120 mg/dl | Yes / No |
| 7 | Resting ECG Results | Normal / ST-T wave abnormality / Left ventricular hypertrophy |
| 8 | Max Heart Rate Achieved | Highest heart rate during a stress test |
| 9 | Exercise-Induced Angina | Chest pain triggered by exercise (Yes/No) |
| 10 | ST Depression (Oldpeak) | ST depression induced by exercise, relative to rest |
| 11 | Slope of Peak Exercise ST Segment | Upsloping / Flat / Downsloping |
| 12 | Major Vessels Colored by Fluoroscopy | Count (0-3) of visible vessels |
| 13 | Thalassemia | Normal / Fixed defect / Reversible defect |
| 14 | *(target, not user input)* | Presence of heart disease — this is what the model predicts |

Every field is validated in the app itself — dropdowns prevent invalid categorical
values, and numeric fields are bounded to physiologically plausible ranges (e.g.
cholesterol 100–600 mg/dl). If a value is out of range or missing, the app blocks
prediction and tells the user exactly what to fix, rather than silently guessing.

## How invalid/missing input is handled

- **In the UI**: Streamlit widgets themselves prevent most bad input (dropdowns for
  categories, min/max-bounded number inputs for continuous values).
- **In code**: `validate_inputs()` in `app.py` re-checks every field before prediction
  runs — defense in depth, in case widget state is ever bypassed.
- **In training data**: the raw UCI dataset has a few missing values (marked `?` in
  the `ca` and `thal` columns). `train_model.py` converts these to `NaN` and imputes
  them (median for numeric, most-frequent for categorical) as part of the pipeline —
  the same imputer is reused automatically at prediction time, so it never crashes on
  an unexpected null.
- **At inference**: an `IsolationForest` anomaly detector flags input combinations
  that look statistically unusual compared to the training data, and the app shows a
  caution banner rather than presenting an overconfident prediction.

## Model training

`train_model.py`:
1. Loads and cleans `data/heart.csv`
2. Builds a preprocessing pipeline (median/mode imputation, scaling, one-hot encoding)
3. Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost** via
   5-fold cross-validated ROC-AUC, and picks the best
4. Wraps the winner in `CalibratedClassifierCV` so the probability shown to users is
   statistically meaningful, not just a raw score
5. Fits an `IsolationForest` for anomaly detection
6. Saves everything (preprocessor + calibrated model + SHAP-ready model + anomaly
   detector + feature metadata) into one `model/heart_model.pkl` via `joblib`

---

## Running locally

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd heart-disease-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install exact pinned dependencies
```bash
pip install -r requirements.txt
```
> Installing from `requirements.txt` *before* training keeps your local scikit-learn
> version identical to the one used to build the committed model — this avoids the
> classic "pickle trained on one sklearn version, loaded on another" crash.

### 3. Train the model
```bash
python train_model.py
```
This creates `model/heart_model.pkl`. You'll see cross-validation scores, the chosen
model, and test-set metrics printed to the console.

### 4. Add your Gemini API key locally
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml` and paste your real key:
```toml
GEMINI_API_KEY = "your-actual-key-here"
```
Get a free key (no credit card) at **https://aistudio.google.com** → "Get API Key".

### 5. Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`).

If you skip step 4, the app still works — it just falls back to showing the raw SHAP
factor breakdown instead of the Gemini plain-language explanation.

---

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Heart disease risk predictor"
git branch -M main
```
Create an empty repository on GitHub (no README/license, to avoid merge conflicts),
then:
```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Before pushing, double check**:
- `.streamlit/secrets.toml` is **not** tracked (`.gitignore` already excludes it —
  verify with `git status`, it should not appear)
- `model/heart_model.pkl` **is** committed — Streamlit Cloud does not run a custom
  build/training step, so the trained model file needs to be in the repo

---

## Deploying to Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Select your repository, branch (`main`), and set the main file path to `app.py`
4. Before clicking Deploy, click **"Advanced settings"** → **Secrets**, and paste:
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   ```
   This is the live equivalent of your local `secrets.toml` — it's stored securely by
   Streamlit Cloud, not in your GitHub repo.
5. Click **Deploy**. Streamlit Cloud will install from `requirements.txt` and launch
   `app.py` automatically.
6. Your app will be live at a URL like:
   `https://<your-app-name>.streamlit.app`

### If you update the model later
Retrain locally (`python train_model.py`), commit the updated `model/heart_model.pkl`,
and push — Streamlit Cloud auto-redeploys on every push to the connected branch.

### Common deploy issues
| Symptom | Likely cause | Fix |
|---|---|---|
| App crashes loading `heart_model.pkl` | scikit-learn/xgboost/shap version mismatch | Confirm `requirements.txt` versions match what you trained with locally (`pip show scikit-learn xgboost shap`) |
| AI explanation never appears | `GEMINI_API_KEY` not set in Streamlit Cloud secrets | Add it under App settings → Secrets, then reboot the app |
| "Model file not found" error | `model/heart_model.pkl` not committed to GitHub | Check `git status` / `.gitignore`, force-add if needed: `git add -f model/heart_model.pkl` |
| Gemini calls fail intermittently | Free tier rate limit (1,500 requests/day) hit | App already falls back to the raw SHAP breakdown automatically — no action needed unless it's persistent |

---

## Tech stack

| Layer | Technology |
|---|---|
| App framework | Streamlit |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| LLM explanation | Google Gemini 2.5 Flash (free tier, no credit card) |
| Model persistence | joblib |
| Hosting | Streamlit Community Cloud |
