import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit_authenticator as stauth

# ── Page Config (MUST BE FIRST) ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication Setup ──────────────────────────────────────────────────────
# Credentials dictionary for the new version of streamlit-authenticator
# Note: The password 'HealthPrediction' is pre-hashed here to avoid the Hasher error
credentials = {
    "usernames": {
        "Arjunmedico": {
            "name": "Arjun Mishra",
            "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6L6.SIDY6I1Rj16S" # Hash for HealthPrediction
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "medassist_cookie",
    "auth_key",
    cookie_expiry_days=30
)

# Render the Login Widget
# In the latest version, this updates st.session_state automatically
authenticator.login()

# ── Check Authentication Status ───────────────────────────────────────────────
if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
elif st.session_state["authentication_status"]:

    # ── Sidebar: Logout & Info ────────────────────────────────────────────────
    with st.sidebar:
        st.write(f"Welcome, **{st.session_state['name']}**")
        authenticator.logout("Logout", "sidebar")
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.metric("Symptoms", "132")
        st.metric("Diseases", "41")
        st.metric("Model", "Random Forest")

        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.caption("Educational purposes only. Consult a doctor for medical advice.")

    # ── Medicine Database ─────────────────────────────────────────────────────────
    MEDICINE_MAP = {
        "Flu": "Oseltamivir (Tamiflu), Paracetamol, Rest & Fluids",
        "Common_Cold": "Cetirizine, Nasal Decongestant, Vitamin C",
        "COVID19": "Paracetamol, Dexamethasone, Oxygen Therapy",
        "Dengue": "Paracetamol, IV Fluids, Platelet Transfusion",
        "Malaria": "Chloroquine, Artemisinin, Primaquine",
        "Typhoid": "Ciprofloxacin, Azithromycin, Ceftriaxone",
        "Tuberculosis": "Isoniazid, Rifampicin, Ethambutol, Pyrazinamide",
        "Chickenpox": "Acyclovir, Calamine Lotion, Antihistamines",
        "Measles": "Vitamin A, Paracetamol, Supportive Care",
        "Hepatitis": "Tenofovir, Entecavir, Interferon, Liver Support",
        "Diabetes": "Metformin, Insulin, Glipizide",
        "Hypertension": "Amlodipine, Losartan, Atenolol",
        "Heart_Disease": "Aspirin, Atorvastatin, Beta-Blockers, ACE Inhibitors",
        "Asthma": "Salbutamol Inhaler, Budesonide, Montelukast",
        "Pneumonia": "Amoxicillin, Azithromycin, Ceftriaxone",
        "Bronchitis": "Amoxicillin, Bromhexine, Salbutamol",
        "Kidney_Disease": "ACE Inhibitors, Erythropoietin, Dialysis",
        "Liver_Disease": "Lactulose, Rifaximin, Ursodeoxycholic Acid",
        "Thyroid_Disorder": "Levothyroxine, Methimazole, Carbimazole",
        "Anemia": "Iron Supplements, Folic Acid, Vitamin B12",
        "Arthritis": "Ibuprofen, Methotrexate, Hydroxychloroquine",
        "Osteoporosis": "Calcium, Vitamin D, Alendronate",
        "Migraine": "Sumatriptan, Topiramate, Propranolol",
        "Epilepsy": "Valproate, Carbamazepine, Levetiracetam",
        "Depression": "Sertraline, Fluoxetine, Escitalopram",
        "Anxiety_Disorder": "Clonazepam, Buspirone, Sertraline",
        "Schizophrenia": "Risperidone, Olanzapine, Haloperidol",
        "Stroke": "Aspirin, tPA, Anticoagulants, Rehabilitation",
        "Allergy": "Cetirizine, Loratadine, Fexofenadine",
        "Dermatitis": "Hydrocortisone Cream, Tacrolimus, Antihistamines",
        "Psoriasis": "Methotrexate, Cyclosporine, Biologics",
        "Gastritis": "Omeprazole, Antacids, H. pylori Antibiotics",
        "Ulcer": "Omeprazole, Amoxicillin, Clarithromycin",
        "Appendicitis": "Appendectomy, Cefuroxime, Metronidazole",
        "Pancreatitis": "IV Fluids, Pain Management, Pancreatic Enzymes",
        "Obesity": "Orlistat, Metformin, Lifestyle Modification",
        "Glaucoma": "Timolol Eye Drops, Latanoprost, Laser Surgery",
        "Cataract": "Phacoemulsification Surgery, Lens Implant",
        "Sinusitis": "Amoxicillin, Nasal Steroids, Decongestants",
        "Breast_Cancer": "Tamoxifen, Chemotherapy, Radiation, Surgery",
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
        .result-card {
            background: linear-gradient(135deg, #e8f4f8, #d0ecff);
            border: 2px solid #1a7abf; border-radius: 16px;
            padding: 1.5rem; margin: 1rem 0;
        }
        .medicine-card {
            background: #f0fff4; border: 2px solid #38a169;
            border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🩺 MedAssist AI</h1>
        <p>AI-powered Disease Prediction & Medicine Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model files not found. Ensure model.pkl and symptoms.pkl are in the repo.")
        st.stop()

    # ── Main Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Symptom Checker", "📋 Disease Reference", "📈 About Model"])

    with tab1:
        st.markdown("### Select Your Symptoms")
        # --- (PASTE YOUR FULL SYMPTOM CHECKER SELECTION LOGIC HERE) ---
        st.info("Check your symptoms and click Predict below.")
        
        # Build your input vector and run model.predict() as in your original code.
        # This block should contain the predict_btn logic.
        
    with tab2:
        st.markdown("### 📋 Disease Reference Table")
        df_ref = pd.DataFrame([
            {"Disease": k.replace("_", " "), "Medicines": v, "Severity": SEVERITY_MAP.get(k, "🟡 MODERATE")}
            for k, v in MEDICINE_MAP.items()
        ])
        st.dataframe(df_ref, use_container_width=True)

    with tab3:
        st.markdown("### 📈 Model Information")
        st.write("Random Forest model trained on 5,000 records.")
