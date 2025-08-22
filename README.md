# Machine Predictive Maintenance

This project implements a **Machine Predictive Maintenance System** using **Machine Learning (Random Forest Classifier)**. The goal is to predict machine failure types based on various sensor readings and machine parameters.

The project provides two deployment options:

1. **Flask Web App** (`app.py`) – Deployed on **Render**.
2. **Streamlit App** (`streamlit.py`) – Deployed on **Streamlit Cloud**.

---

## 📂 Project Structure

```
├── predictive_maintenance.csv       # Dataset
├── predictive_maintenance.pkl       # Trained Random Forest model
├── scaler.pkl                       # StandardScaler used for preprocessing
├── app.py                           # Flask application
├── streamlit.py                     # Streamlit application
├── templates/
│   └── index.html                   # HTML template for Flask app
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 📊 Dataset

The dataset contains machine sensor readings and failure labels. Some important features include:

- **Type** → Machine type (`L`, `M`, `H`)
- **Air temperature [K]**
- **Process temperature [K]**
- **Rotational speed [rpm]**
- **Torque [Nm]**
- **Tool wear [min]**
- **Target** → Whether machine failed (1 = failure, 0 = normal)
- **Failure Type** → Specific failure type when `Target = 1`

---

## ⚙️ Model Training Pipeline

1. **Exploratory Data Analysis (EDA)**: Distribution plots, correlations, skewness check.
2. **Preprocessing**:
   - Encode categorical `Type` using `OrdinalEncoder` (`L=0, M=1, H=2`).
   - Scale numerical features (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`) using `StandardScaler`.
3. **Modeling**:
   - Used `RandomForestClassifier`.
   - Hyperparameter tuning with `GridSearchCV`.
   - Saved best model as `predictive_maintenance.pkl`.
   - Saved scaler as `scaler.pkl`.

---

## 🚀 Running the Project Locally

### 🔹 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/<your-username>/machine-predictive-maintenance.git
cd machine-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 🔹 2. Run Flask App

```bash
python app.py
```

The app will be available at local server: [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/)

### 🔹 3. Run Streamlit App

```bash
streamlit run streamlit.py
```

The app will be available at local server: [**http://localhost:8501/**](http://localhost:8501/)

---

## 🌍 Deployed Versions

- **Streamlit App** → [https://machine-predictivemaintenance-rycn3kobq3x3jefkrd9hak.streamlit.app/](https://machine-predictivemaintenance-rycn3kobq3x3jefkrd9hak.streamlit.app/)
- **Flask App (Render)** → [https://machine-predictive-maintenance-aofj.onrender.com](https://machine-predictive-maintenance-aofj.onrender.com)

---

## 🖥️ Flask App Workflow

1. User submits form with machine parameters.
2. Data is preprocessed (encoding + scaling).
3. Trained model predicts failure type.
4. Prediction displayed on webpage.

---

## 🎛️ Streamlit App Workflow

1. User selects **Product Type** and adjusts sliders for sensor readings.
2. Input features are preprocessed using saved scaler.
3. Model predicts failure type.
4. Result displayed interactively in Streamlit UI.

---

## 📈 Model Evaluation

- **Algorithm**: Random Forest Classifier
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- flask
- streamlit

Install them with:

```bash
pip install -r requirements.txt
```

---

## ✨ Future Improvements

- Add **real-time sensor data streaming** (Kafka, MQTT).
- Deploy model to **cloud (AWS/GCP/Azure)**.
- Improve model with **deep learning (LSTMs for time-series)**.
- Build CI/CD pipeline for continuous deployment.

