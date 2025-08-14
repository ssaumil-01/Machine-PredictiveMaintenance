# Machine Predictive Maintenance — Failure Type Classification (FastAPI + XGBoost)

Predict the **type of failure** a machine is likely to experience using a production-grade ML pipeline deployed behind a **FastAPI** service.

---

## ✨ Highlights

- **Clear problem framing**: Multiclass classification of *failure type* (trained **only** on rows where `machine failure == 1` to avoid leakage).
- **Solid pipeline**: Feature engineering (quality extraction from `productID`), scaling, one-hot encoding, and class-imbalance handling.
- **Strong results**: ~**0.89** accuracy and ~**0.86** macro F1 on the test set.
- **Deployment ready**: FastAPI app with a clean `/predict` endpoint and a `pydantic` schema; model serialized via `joblib` and served with Uvicorn.
- **Reproducible**: Single source of truth for preprocessing used consistently in training and inference.

---

## 🧠 Problem Statement

Predict the **failure type** among multiple categories for each machine record. The training data uses **only records where `machine failure = 1`**; the `machine failure` column is dropped before training to prevent leakage. This focuses the classifier on separating failure **subtypes** rather than failure vs. no-failure.

---

## 🗂️ Data Overview

- **Source**: Predictive maintenance classification dataset (10,000 rows, 14 columns).
- **Core features** (examples):  
  - `air temperature [K]`, `process temperature [K]`  
  - `rotational speed [rpm]`, `torque [Nm]`  
  - `tool wear [min]`  
  - `productID` → contains embedded quality code (`L`, `M`, `H`) plus serial
- **Target**: `failure type` (multiclass)
- **Note**: `machine failure` is used only to select failure rows (==1) and then **removed** prior to training.

During EDA, class imbalance across failure categories was observed; sensor variables show meaningful relationships with failure subtypes, and **tool wear** emerges as a strong signal. Quality levels (`L/M/H`) show different patterns w.r.t. `torque` and `speed`.

---

## 🏗️ Pipeline & Modeling

### Preprocessing
- **Quality feature**: Extract categorical `quality` from `productID` → one-hot encode (`L/M/H`).
- **Numeric features**: Standardize (`StandardScaler`) for temperature, speed, torque, wear.
- **Train/test split**: Standard holdout evaluation.
- **Imbalance mitigation**: SMOTE and/or classifier `class_weight` for minority classes.

### Models Evaluated
- `RandomForestClassifier`
- `XGBoostClassifier`  ✅ **Selected** (best overall performance after tuning)

### Performance (Test Set)
| Metric | Score (approx.) |
|---|---|
| Accuracy | **0.89** |
| F1 (macro) | **0.86** |

Confusion matrices and per-class precision/recall were inspected to understand misclassifications and residual class imbalance.

> Tip: Keep inference-time preprocessing **identical** to training (encode `quality` exactly as during training; use the same scaler/encoder artifacts).

---

## 🖥️ API (FastAPI)

### Stack
- **API**: FastAPI + Uvicorn
- **ML**: scikit-learn (preprocessing), XGBoost (model), imbalanced-learn (SMOTE)
- **Serialization**: `joblib`
- **Validation**: `pydantic` for request schema

### Endpoint
- `POST /predict` → returns the predicted failure subtype

#### Example Request Body
```json
{
  "air_temperature": 300,
  "process_temperature": 310,
  "rotational_speed": 1600,
  "torque": 45,
  "tool_wear": 200,
  "product_quality_L": 0,
  "product_quality_M": 1,
  "product_quality_H": 0
}
```

#### Example Response
```json
{
  "predicted_failure_type": "Tool Wear Failure"
}
```

> If your service prefers deriving quality from `productID` on the fly, accept `productID` and parse `L/M/H` server-side, but ensure **training-time** and **serving-time** logic match.

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# Minimal set if you don't have a lockfile:
pip install fastapi uvicorn scikit-learn xgboost imbalanced-learn pandas numpy joblib pydantic
```

### 3) Train (illustrative sketch)
```python
# scripts/train.py (example outline)
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier

# 1) Load data
df = pd.read_csv("data/train.csv")

# 2) Filter failures only and prepare target
df = df[df["machine failure"] == 1].copy()
y = df["failure type"]
X = df.drop(columns=["failure type", "machine failure"])

# 3) Extract quality from productID → one-hot
X["quality"] = X["productID"].str[0]
X = X.drop(columns=["productID"])

num_cols = ["air temperature [K]","process temperature [K]","rotational speed [rpm]","torque [Nm]","tool wear [min]"]
cat_cols = ["quality"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", clf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("Acc:", accuracy_score(y_test, y_pred))
print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))

joblib.dump(pipe, "model/model.joblib")
```

### 4) Run the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5) Call the endpoint
```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{"air_temperature":300,"process_temperature":310,"rotational_speed":1600,"torque":45,"tool_wear":200,"product_quality_L":0,"product_quality_M":1,"product_quality_H":0}'
```

---

## 📦 Suggested Repo Structure

```
.
├─ main.py                 # FastAPI app (loads model, defines /predict)
├─ model/
│  └─ model.joblib         # serialized pipeline (preprocessing + classifier)
├─ utils/
│  └─ preprocessing.py     # (optional) shared feature logic
├─ scripts/
│  └─ train.py             # training entrypoint
├─ data/                   # raw/processed data (keep small samples in repo)
├─ tests/
│  └─ test_api.py          # simple e2e test for /predict
├─ requirements.txt
└─ README.md
```

---

## 🧪 Testing

- **Unit tests** for preprocessing helpers (e.g., quality extraction).
- **E2E tests** for `/predict` using `httpx` or `requests` and a small fixture.
- Ensure the serialized pipeline used in API matches the training artifact.

---

## 🔒 Responsible Use & Limitations

- Model is trained on failure-only rows; it **does not** decide whether a failure will occur—only **which type** given a failure context.
- Handle class imbalance carefully; monitor per-class recall (especially minority classes).
- Inputs must respect training-time ranges/units; out-of-distribution values degrade accuracy.
- Consider adding probability thresholds, abstention, and monitoring in production.

---

## 🛣️ Roadmap Ideas

- Calibrated probabilities & uncertainty estimates
- SHAP-based explanations in the API
- Periodic retraining + data drift monitoring
- Dockerfile & CI/CD (GitHub Actions)
- Feature store for consistent `productID → quality` extraction

---

## 📄 License

Add a license (e.g., MIT) to clarify usage.

---

## 🙌 Acknowledgments

- Dataset and original task framing credit to the predictive maintenance dataset authors/community.
