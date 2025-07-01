# 🧪 Compliance Classifier

A machine learning model that predicts whether a cosmetic product formulation is compliant or non-compliant based on its INCI ingredient list.

## 💡 Overview

- Uses TF-IDF vectorization to transform ingredient lists
- Trains a Logistic Regression and Random Forest classifier
- Evaluates with precision, recall, f1-score, and confusion matrix
- Designed for regulatory teams, formulation scientists, and AI product prototyping

## 📦 Files

- `logistic_compliance_classifier.py` – Baseline logistic regression model
- `random_forest_compliance_classifier.py` – Enhanced Random Forest version
- `compliance_dataset.csv` – Simulated dataset with compliance labels
- `requirements.txt` – Install dependencies

## ▶️ Run the Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run either script:

```bash
python logistic_compliance_classifier.py
# or
python random_forest_compliance_classifier.py
```

## 🧠 Dataset

A synthetic dataset of 100 cosmetic products labeled as compliant or non-compliant based on the presence of common regulatory-risk ingredients such as parabens, triclosan, or formaldehyde.

## ✍️ Author

Built by Sky High AI for internal ML exploration and portfolio readiness.
