# 🩺 MedAssist AI — Setup & Run Guide

## What's Included
| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `train_model.py` | One-time model training script |
| `requirements.txt` | Python dependencies |
| `healthcare_dataset_5000_people.csv` | Your dataset (copy here) |

---

## Step-by-Step Setup

### 1. Copy Your Dataset
Place `healthcare_dataset_5000_people.csv` in the same folder as these files:
```
Documents/MedAssist_AI/
├── app.py
├── train_model.py
├── requirements.txt
└── healthcare_dataset_5000_people.csv   ← copy here
```

### 2. Install Dependencies
Open terminal / command prompt in the MedAssist_AI folder:
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Run Once)
```bash
python train_model.py
```
This creates `model.pkl` and `symptoms.pkl`.
Expected output:
```
✅ Accuracy: ~95%+
✅ Saved: model.pkl
✅ Saved: symptoms.pkl
🚀 Ready! Now run: streamlit run app.py
```

### 4. Launch the App
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

---

## Features
- ✅ 132 symptom checkboxes grouped by body system
- ✅ Real-time disease prediction with confidence %
- ✅ Top 3 most likely diseases shown
- ✅ Medicine recommendation for all 41 diseases
- ✅ Severity alerts (Urgent / High / Moderate / Mild)
- ✅ Disease & Medicine reference table with search
- ✅ Model information dashboard

## 41 Diseases Covered
Flu, COVID-19, Dengue, Malaria, Typhoid, Tuberculosis, Pneumonia, Asthma,
Bronchitis, Heart Disease, Hypertension, Diabetes, Thyroid Disorder, Kidney Disease,
Liver Disease, Hepatitis, Anemia, Arthritis, Osteoporosis, Migraine, Epilepsy,
Stroke, Depression, Anxiety Disorder, Schizophrenia, Allergy, Dermatitis,
Psoriasis, Gastritis, Ulcer, Appendicitis, Pancreatitis, Obesity, Glaucoma,
Cataract, Sinusitis, Chickenpox, Measles, Breast Cancer, Prostate Disorder, Common Cold

---

## ⚠️ Disclaimer
This system is for **educational purposes only**.
Always consult a qualified doctor before taking any medication.
