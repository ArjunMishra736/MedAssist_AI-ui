"""
MedAssist AI — Streamlit App
==============================
Disease Prediction + Medicine Recommendation System
Powered by Random Forest (132 symptoms → 41 diseases)

Run:
    streamlit run app.py
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Medicine Database ─────────────────────────────────────────────────────────
MEDICINE_MAP = {
    "Flu":               "Oseltamivir (Tamiflu), Paracetamol, Rest & Fluids",
    "Common_Cold":       "Cetirizine, Nasal Decongestant, Vitamin C",
    "COVID19":           "Paracetamol, Dexamethasone, Oxygen Therapy",
    "Dengue":            "Paracetamol, IV Fluids, Platelet Transfusion",
    "Malaria":           "Chloroquine, Artemisinin, Primaquine",
    "Typhoid":           "Ciprofloxacin, Azithromycin, Ceftriaxone",
    "Tuberculosis":      "Isoniazid, Rifampicin, Ethambutol, Pyrazinamide",
    "Chickenpox":        "Acyclovir, Calamine Lotion, Antihistamines",
    "Measles":           "Vitamin A, Paracetamol, Supportive Care",
    "Hepatitis":         "Tenofovir, Entecavir, Interferon, Liver Support",
    "Diabetes":          "Metformin, Insulin, Glipizide",
    "Hypertension":      "Amlodipine, Losartan, Atenolol",
    "Heart_Disease":     "Aspirin, Atorvastatin, Beta-Blockers, ACE Inhibitors",
    "Asthma":            "Salbutamol Inhaler, Budesonide, Montelukast",
    "Pneumonia":         "Amoxicillin, Azithromycin, Ceftriaxone",
    "Bronchitis":        "Amoxicillin, Bromhexine, Salbutamol",
    "Kidney_Disease":    "ACE Inhibitors, Erythropoietin, Dialysis",
    "Liver_Disease":     "Lactulose, Rifaximin, Ursodeoxycholic Acid",
    "Thyroid_Disorder":  "Levothyroxine, Methimazole, Carbimazole",
    "Anemia":            "Iron Supplements, Folic Acid, Vitamin B12",
    "Arthritis":         "Ibuprofen, Methotrexate, Hydroxychloroquine",
    "Osteoporosis":      "Calcium, Vitamin D, Alendronate",
    "Migraine":          "Sumatriptan, Topiramate, Propranolol",
    "Epilepsy":          "Valproate, Carbamazepine, Levetiracetam",
    "Depression":        "Sertraline, Fluoxetine, Escitalopram",
    "Anxiety_Disorder":  "Clonazepam, Buspirone, Sertraline",
    "Schizophrenia":     "Risperidone, Olanzapine, Haloperidol",
    "Stroke":            "Aspirin, tPA, Anticoagulants, Rehabilitation",
    "Allergy":           "Cetirizine, Loratadine, Fexofenadine",
    "Dermatitis":        "Hydrocortisone Cream, Tacrolimus, Antihistamines",
    "Psoriasis":         "Methotrexate, Cyclosporine, Biologics",
    "Gastritis":         "Omeprazole, Antacids, H. pylori Antibiotics",
    "Ulcer":             "Omeprazole, Amoxicillin, Clarithromycin",
    "Appendicitis":      "Appendectomy, Cefuroxime, Metronidazole",
    "Pancreatitis":      "IV Fluids, Pain Management, Pancreatic Enzymes",
    "Obesity":           "Orlistat, Metformin, Lifestyle Modification",
    "Glaucoma":          "Timolol Eye Drops, Latanoprost, Laser Surgery",
    "Cataract":          "Phacoemulsification Surgery, Lens Implant",
    "Sinusitis":         "Amoxicillin, Nasal Steroids, Decongestants",
    "Breast_Cancer":     "Tamoxifen, Chemotherapy, Radiation, Surgery",
    "Prostate_Disorder": "Tamsulosin, Finasteride, Dutasteride",
}

SEVERITY_MAP = {
    "Appendicitis": "🔴 URGENT", "Stroke": "🔴 URGENT",
    "Heart_Disease": "🔴 HIGH", "Breast_Cancer": "🔴 HIGH",
    "Tuberculosis": "🟠 MODERATE-HIGH", "COVID19": "🟠 MODERATE-HIGH",
    "Pneumonia": "🟠 MODERATE-HIGH", "Dengue": "🟠 MODERATE-HIGH",
    "Malaria": "🟡 MODERATE", "Typhoid": "🟡 MODERATE",
    "Flu": "🟢 MILD-MODERATE", "Common_Cold": "🟢 MILD",
    "Sinusitis": "🟢 MILD", "Allergy": "🟢 MILD",
}

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        symptoms = pickle.load(open("symptoms.pkl", "rb"))
        return model, symptoms
    except FileNotFoundError:
        return None, None

model, SYMPTOMS = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f4c81 0%, #1a7abf 50%, #00a8e8 100%);
        padding: 2rem; border-radius: 16px; color: white;
        margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { font-size: 2.8rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.9; }
    
    .result-card {
        background: linear-gradient(135deg, #e8f4f8, #d0ecff);
        border: 2px solid #1a7abf; border-radius: 16px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .medicine-card {
        background: #f0fff4; border: 2px solid #38a169;
        border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0;
    }
    .warning-card {
        background: #fff8e1; border: 2px solid #f59e0b;
        border-radius: 12px; padding: 1rem; margin: 1rem 0;
    }
    .stat-card {
        background: white; border-radius: 12px;
        padding: 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1a7abf !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 MedAssist AI</h1>
    <p>AI-powered Disease Prediction & Medicine Recommendation System</p>
    <p><small>132 Symptoms • 41 Diseases • 5,000 Patient Dataset • Random Forest</small></p>
</div>
""", unsafe_allow_html=True)

# ── Model Not Found Warning ───────────────────────────────────────────────────
if model is None:
    st.error("""
    ⚠️ **Model not found!** Run this first in your terminal:
    ```
    python train_model.py
    ```
    Then restart the app.
    """)
    st.stop()

# ── Sidebar: Stats ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 System Info")
    st.metric("Symptoms", "132")
    st.metric("Diseases", "41")
    st.metric("Training Samples", "4,000")
    st.metric("Model", "Random Forest")
    st.metric("Trees", "200")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This AI system is for **educational purposes only**. "
        "Always consult a qualified doctor before taking any medication. "
        "Do not use this as a substitute for professional medical advice."
    )

    st.markdown("---")
    st.markdown("### 🏥 Emergency")
    st.error("🚨 Life-threatening? Call **112** immediately")

# ── Main Area ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Symptom Checker", "📋 Disease Reference", "📈 About Model"])

# ── TAB 1: Symptom Checker ───────────────────────────────────────────────────
with tab1:
    st.markdown("### Select Your Symptoms")
    st.caption("Choose all symptoms you are currently experiencing:")

    # Group symptoms into categories for better UX
    symptom_categories = {
        "🌡️ General": ["fever", "fatigue", "chills", "weakness", "weight_loss", "weight_gain", "night_sweats"],
        "🫁 Respiratory": ["cough", "shortness_of_breath", "sore_throat", "runny_nose", "nasal_congestion",
                           "post_nasal_drip", "hoarseness", "wheezing", "chest_tightness", "cough_blood",
                           "difficulty_breathing", "rapid_breathing", "slow_breathing", "loss_of_smell"],
        "🤢 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "constipation", "bloating",
                         "gas", "heartburn", "acid_reflux", "loss_of_appetite", "difficulty_swallowing",
                         "blood_in_stool", "pale_stool"],
        "❤️ Cardiovascular": ["chest_pain", "palpitations", "high_bp", "low_bp", "irregular_heartbeat",
                               "fainting", "swelling"],
        "🧠 Neurological": ["headache", "dizziness", "confusion", "memory_loss", "tremors", "numbness",
                             "tingling", "seizures", "speech_difficulty", "loss_of_balance",
                             "difficulty_walking", "migraine", "photophobia", "phonophobia"],
        "😴 Mental Health": ["anxiety", "depression", "insomnia", "restlessness", "panic_attacks",
                              "irritability", "mood_swings", "hallucinations", "delusions",
                              "paranoia", "poor_concentration"],
        "👁️ Eyes & ENT": ["blurred_vision", "eye_redness", "light_sensitivity", "eye_discharge",
                           "vision_loss", "ear_pain", "hearing_loss", "ringing_ears"],
        "🦴 Musculoskeletal": ["muscle_pain", "joint_pain", "back_pain", "neck_pain", "shoulder_pain",
                                "hip_pain", "knee_pain", "ankle_pain", "finger_pain", "hand_stiffness",
                                "bone_pain", "leg_cramps"],
        "🩺 Metabolic": ["frequent_urination", "thirst", "dry_mouth", "high_blood_sugar",
                          "low_blood_sugar", "excess_hunger", "heat_intolerance", "cold_intolerance"],
        "🎗️ Skin": ["skin_rash", "itching", "dry_skin", "oily_skin", "yellow_skin", "excess_sweating",
                     "hair_loss", "skin_ulcers", "slow_healing_wounds", "lump"],
        "🔬 Other": ["yellow_eyes", "dark_urine", "blood_in_urine", "painful_urination",
                      "urinary_retention", "urinary_incontinence", "cold_hands", "cold_feet",
                      "foot_swelling", "frequent_infections", "gum_bleeding", "bad_breath",
                      "mouth_ulcers", "tooth_pain", "snoring", "sleep_apnea",
                      "heavy_menstruation", "missed_periods", "pelvic_pain", "breast_pain",
                      "breast_discharge", "erectile_dysfunction", "testicular_pain",
                      "scrotal_swelling", "developmental_delay", "growth_delay",
                      "learning_difficulty", "loss_of_taste"]
    }

    selected_symptoms = []
    cols = st.columns(2)

    for i, (category, syms) in enumerate(symptom_categories.items()):
        # Only show symptoms that exist in our model
        valid_syms = [s for s in syms if s in SYMPTOMS]
        if not valid_syms:
            continue
        with cols[i % 2]:
            with st.expander(category, expanded=(i < 2)):
                for sym in valid_syms:
                    if st.checkbox(sym.replace("_", " ").title(), key=sym):
                        selected_symptoms.append(sym)

    st.markdown("---")

    # Selected symptoms summary
    if selected_symptoms:
        st.info(f"**{len(selected_symptoms)} symptoms selected:** " +
                ", ".join([s.replace('_', ' ').title() for s in selected_symptoms]))

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        predict_btn = st.button("🔍 Predict Disease", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Clear All", use_container_width=True):
            st.rerun()

    # ── Prediction ───────────────────────────────────────────────────────────
    if predict_btn:
        if len(selected_symptoms) < 2:
            st.warning("⚠️ Please select at least 2 symptoms for an accurate prediction.")
        else:
            # Build input vector
            input_vector = [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
            input_array = np.array(input_vector).reshape(1, -1)

            # Predict with probabilities
            prediction = model.predict(input_array)[0]
            probabilities = model.predict_proba(input_array)[0]
            classes = model.classes_

            # Top 3 predictions
            top3_idx = np.argsort(probabilities)[::-1][:3]
            top3 = [(classes[i], probabilities[i]) for i in top3_idx]

            confidence = top3[0][1] * 100

            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")

            # Primary result
            severity = SEVERITY_MAP.get(prediction, "🟡 MODERATE")
            st.markdown(f"""
            <div class="result-card">
                <h2>🦠 Predicted Disease: <strong>{prediction.replace('_', ' ')}</strong></h2>
                <p><strong>Confidence:</strong> {confidence:.1f}% &nbsp;|&nbsp;
                   <strong>Severity:</strong> {severity}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.progress(confidence / 100)

            # Top 3 alternatives
            st.markdown("### 🔬 Top 3 Predictions")
            for rank, (disease, prob) in enumerate(top3, 1):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(prob, text=f"{rank}. {disease.replace('_', ' ')}")
                with col_b:
                    st.metric("", f"{prob*100:.1f}%")

            # Medicine recommendation
            medicine = MEDICINE_MAP.get(prediction, "Consult a doctor for specific medication")
            st.markdown(f"""
            <div class="medicine-card">
                <h3>💊 Recommended Medicines</h3>
                <p style="font-size:1.1rem"><strong>{medicine}</strong></p>
                <small>Note: Always take medicines under doctor's supervision</small>
            </div>
            """, unsafe_allow_html=True)

            # Urgent warning for serious diseases
            if severity.startswith("🔴"):
                st.error("🚨 **URGENT**: This condition requires immediate medical attention! "
                         "Visit a hospital or call emergency services NOW.")
            elif severity.startswith("🟠"):
                st.warning("⚠️ **Please consult a doctor soon.** This condition needs professional evaluation.")

            # Disclaimer
            st.markdown("""
            <div class="warning-card">
                ⚠️ <strong>Medical Disclaimer:</strong> This prediction is generated by an AI model 
                trained on 5,000 patient records. It is for informational purposes only and 
                <strong>NOT a substitute for professional medical diagnosis.</strong> 
                Always consult a qualified healthcare professional.
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Disease Reference ─────────────────────────────────────────────────
with tab2:
    st.markdown("### 📋 Disease & Medicine Reference Table")
    df_ref = pd.DataFrame([
        {"Disease": k.replace("_", " "), "Medicines": v, "Severity": SEVERITY_MAP.get(k, "🟡 MODERATE")}
        for k, v in MEDICINE_MAP.items()
    ])
    search = st.text_input("🔍 Search disease or medicine...", "")
    if search:
        mask = (df_ref["Disease"].str.contains(search, case=False) |
                df_ref["Medicines"].str.contains(search, case=False))
        df_ref = df_ref[mask]
    st.dataframe(df_ref, use_container_width=True, height=500)

# ── TAB 3: About Model ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm", "Random Forest")
    col2.metric("Trees", "200")
    col3.metric("Diseases", "41")
    col4.metric("Symptoms", "132")

    st.markdown("""
    #### How It Works
    1. **You select symptoms** from 132 binary symptom features
    2. **Random Forest** evaluates 200 decision trees in parallel
    3. **Majority vote** from all trees determines the predicted disease
    4. **Confidence score** reflects the proportion of trees that agreed
    5. **Medicine recommendation** is looked up from a curated database

    #### Dataset
    - **5,000 patient records** with real-world symptom patterns
    - **41 disease categories** covering common to complex conditions
    - **Train/Test split**: 80% / 20% (4000 train, 1000 test)
    
    #### Why Random Forest?
    - Handles high-dimensional data (132 features) extremely well
    - Resistant to overfitting via ensemble averaging
    - Provides feature importance — most diagnostic symptoms ranked
    - No need for feature scaling or normalization
    - Fast prediction time in production
    """)
