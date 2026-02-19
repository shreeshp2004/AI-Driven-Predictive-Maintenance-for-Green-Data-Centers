# 🌱 AI-Driven Predictive Maintenance for Green Data Centers

A machine learning framework for **energy consumption forecasting and carbon footprint analysis** using time-series feature engineering and Random Forest optimization.

---

## 🚀 Overview

This project builds an intelligent energy analytics pipeline that:

* Predicts data center power consumption
* Estimates carbon emissions
* Detects high vs low energy usage patterns
* Optimizes model performance using GridSearchCV

The system is designed to support **green data center initiatives** by enabling data-driven energy optimization and sustainability monitoring.

---

## 🎯 Key Features

✅ Time-series feature engineering from datetime
✅ Random Forest regression with hyperparameter tuning
✅ Energy → Carbon emission estimation
✅ Binary energy classification (High vs Low usage)
✅ Confusion matrix visualization
✅ Model persistence using Joblib
✅ Multiple performance metrics

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

* Convert `Datetime` to pandas datetime
* Extract temporal features:

  * Year
  * Month
  * Day
  * Hour
  * DayOfWeek
* Handle missing values
* Train-test split (80/20)

---

### 2️⃣ Model Training

Model used:

```
RandomForestRegressor
```

Hyperparameter tuning via:

```
GridSearchCV
```

Parameters optimized:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

---

### 3️⃣ Evaluation Metrics

#### 🔹 Regression Metrics

* MAE
* MSE
* RMSE
* R² Score

#### 🔹 Classification Metrics

(derived using median threshold)

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📊 Visualizations Generated

The pipeline automatically produces:

* 📈 Energy consumption trend
* 🌍 Carbon emission trend
* 🔲 Confusion matrix
* 📉 Training vs validation curves

---

## 📁 Dataset

Expected file:

```
AEP_hourly.csv
```

Required columns:

* Datetime
* AEP_MW

Place the dataset in the project root before running.

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

After execution, the model will be saved as:

```
optimized_carbon_footprint_model.pkl
```

---

## 🧪 Example Output

The script prints:

* Regression performance
* Classification performance
* Confusion matrix
* Graphical analysis

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## 📦 Model File

Trained model:

```
optimized_carbon_footprint_model.pkl
```

> ⚠️ Large model files are tracked using Git LFS (recommended).

---

## 🌍 Sustainability Impact

This work supports:

* Green data center optimization
* Carbon footprint awareness
* Energy-efficient infrastructure planning
* AI-driven sustainability analytics

---

## 👨‍💻 Author

**Shreesh Prateek Pathak**
B.Tech ECE (Biomedical Specialization)
VIT Vellore

---

## 📜 License

MIT License

---

## ⭐ Future Improvements

* Federated learning integration
* Real-time IoT data ingestion
* Deep learning models (LSTM/Transformer)
* Explainable AI (XAI)
* Deployment dashboard
