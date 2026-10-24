# 🚦 Traffic Violation Prediction System

A web-based machine learning application that predicts **traffic violation enforcement priority** — Low, Medium, or High — from real-world features like fine amount, violation severity, payment compliance, and time-of-day risk patterns.

Built with **Flask** (backend) and **HTML/CSS** (frontend), powered by a **Gradient Boosting Classifier** selected after benchmarking against Logistic Regression and Random Forest.

---

## Features

- Predicts enforcement priority (Low / Medium / High) from 7 violation features
- Shows **model confidence** alongside each prediction
- Clean, responsive UI
- Reproducible training pipeline with model benchmarking and full metrics
- Production-ready deployment config (gunicorn + Procfile)

---

## Model

The deployed model is a **Gradient Boosting Classifier** (scikit-learn), chosen because it gave the best macro-F1 across three candidates:

| Model                | Role                      |
|----------------------|---------------------------|
| Logistic Regression  | Baseline (with scaling)   |
| Random Forest        | Ensemble benchmark        |
| Gradient Boosting    | **Selected model**        |

Run `train.py` to reproduce the comparison and regenerate `model.pkl`.

**Input features:** `Fine_Amount_PKR`, `Hour_of_Day`, `Time_Period`, `Violation_Level`, `Payment_Compliance`, `Total_Violations`, `Total_Unpaid`

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/mahamrizwan/traffic-violation-prediction-system.git
cd traffic-violation-prediction-system

# 2. Install
pip install -r requirements.txt

# 3. Run
python app.py
# open http://127.0.0.1:5000
```

### Retrain the model
```bash
python train.py --data traffic_violations.csv
```

### Deploy (Render / Railway / Heroku)
The included `Procfile` starts the app with gunicorn:
```
web: gunicorn app:app
```

---

## Tech Stack

Python · Flask · scikit-learn · pandas · NumPy · gunicorn · HTML/CSS

---

## Project Structure

```
├── app.py              # Flask app + prediction endpoint
├── train.py            # Model benchmarking + training
├── model.pkl           # Saved Gradient Boosting model
├── requirements.txt
├── Procfile            # Production start command
├── templates/          # index.html, predict.html
└── static/             # style.css, assets
```
