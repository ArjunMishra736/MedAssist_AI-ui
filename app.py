import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit_authenticator as stauth

# ── 1. Page Config (Must be first) ───────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Authentication Setup ──────────────────────────────────────────────────
# This internal generation ensures the hash always matches your input
names = ["Arjun Mishra"]
usernames = ["Arjunmedico"]
passwords = ["HealthPrediction"]

hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    "usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
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
# Note: Newer versions use login() without arguments, 
# but we add them for maximum compatibility.
try:
    authenticator.login()
except:
    authenticator.login('Login', 'main')

# ── 3. Check Authentication Status ───────────────────────────────────────────
if st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
elif st.session_state.get("authentication_status") is None:
    st.warning('Please enter your username and password')
elif st.session_state.get("authentication_status"):

    # ── 4. Medicine & Severity Maps ──────────────────────────────────────────
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

    # ── 5. Load Model ────────────────────────────────────────────────────────
    @st.cache_resource
    def load_model():
        try:
            model = pickle.load(open("model.pkl", "rb"))
            symptoms = pickle.load(open("symptoms.pkl", "rb"))
            return model, symptoms
        except:
            return None, None

    model, SYMPTOMS = load_model()

    # ── 6. Custom CSS ────────────────────────────────────────────────────────
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
            border: 2px solid #1a7abf; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        }
        .medicine-card {
            background: #f0fff4; border: 2px solid #38a169;
            border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── 7. Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🩺 MedAssist AI</h1>
        <p>AI-powered Disease Prediction & Medicine Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 8. Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state['name']}")
        authenticator.logout("Logout", "sidebar")
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.metric("Symptoms", "132")
        st.metric("Diseases", "41")
        st.metric("Model", "Random Forest")
        st.markdown("---")
        st.error("🚨 Emergency? Call **112**")

    # ── 9. Main Tabs & Symptom Logic ─────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Symptom Checker", "📋 Disease Reference", "📈 About Model"])

    with tab1:
        st.markdown("### Select Your Symptoms")
        
        # CATEGORIES
        symptom_categories = {
            "🌡️ General": ["fever", "fatigue", "chills", "weakness"],
            "🫁 Respiratory": ["cough", "shortness_of_breath", "sore_throat", "runny_nose"],
            "🤢 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
            "🧠 Neurological": ["headache", "dizziness", "confusion"],
            # (You can add more symptoms here following the SYMPTOMS list)
        }

        selected_symptoms = []
        cols = st.columns(2)

        for i, (category, syms) in enumerate(symptom_categories.items()):
            valid_syms = [s for s in syms if s in SYMPTOMS]
            with cols[i % 2]:
                with st.expander(category, expanded=True):
                    for sym in valid_syms:
                        if st.checkbox(sym.replace("_", " ").title(), key=sym):
                            selected_symptoms.append(sym)

        if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
            if len(selected_symptoms) < 2:
                st.warning("⚠️ Select at least 2 symptoms.")
            elif model:
                # Build Vector
                input_v = [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
                pred = model.predict(np.array(input_v).reshape(1, -1))[0]
                
                st.markdown(f"""<div class="result-card">
                    <h2>🦠 Result: {pred.replace('_', ' ')}</h2>
                    <p>Severity: {SEVERITY_MAP.get(pred, '🟡 MODERATE')}</p>
                </div>""", unsafe_allow_html=True)
                
                med = MEDICINE_MAP.get(pred, "Consult a doctor.")
                st.markdown(f"""<div class="medicine-card">
                    <h3>💊 Recommended:</h3><p>{med}</p>
                </div>""", unsafe_allow_html=True)

    with tab2:
        df_ref = pd.DataFrame([{"Disease": k, "Medicine": v} for k,v in MEDICINE_MAP.items()])
        st.dataframe(df_ref, use_container_width=True)

    with tab3:
        st.write("Model: Random Forest | Accuracy: High | Dataset: 5,000 samples")
