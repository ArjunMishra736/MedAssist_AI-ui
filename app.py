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
# Pre-calculated Hash for "HealthPrediction" to avoid the Hasher TypeError
hashed_pw = '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6L6.SIDY6I1Rj16S'

credentials = {
    "usernames": {
        "Arjunmedico": {
            "name": "Arjun Mishra",
            "password": hashed_pw
        }
    }
}

# Create the authenticator object
authenticator = stauth.Authenticate(
    credentials,
    "medassist_cookie",
    "auth_key",
    cookie_expiry_days=30
)

# Render the Login Widget
# Note: Newer versions use login() without arguments
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

    # ── 4. Medicine & Severity Database ──────────────────────────────────────
    MEDICINE_MAP = {
        "Flu": "Oseltamivir (Tamiflu), Paracetamol, Rest & Fluids",
        "Common_Cold": "Cetirizine, Nasal Decongestant, Vitamin C",
        "COVID19": "Paracetamol, Dexamethasone, Oxygen Therapy",
        "Dengue": "Paracetamol, IV Fluids, Platelet Transfusion",
        "Malaria": "Chloroquine, Artemisinin, Primaquine",
        "Typhoid": "Ciprofloxacin, Azithromycin, Ceftriaxone",
        "Diabetes": "Metformin, Insulin, Glipizide",
        "Hypertension": "Amlodipine, Losartan, Atenolol",
        "Heart_Disease": "Aspirin, Atorvastatin, Beta-Blockers",
        "Asthma": "Salbutamol Inhaler, Budesonide, Montelukast",
        "Allergy": "Cetirizine, Loratadine, Fexofenadine",
    }

    SEVERITY_MAP = {
        "Heart_Disease": "🔴 HIGH", "COVID19": "🟠 MODERATE-HIGH",
        "Flu": "🟢 MILD-MODERATE", "Common_Cold": "🟢 MILD",
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

    # ── 6. Custom Styling ────────────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
        .main-header {
            background: linear-gradient(135deg, #0f4c81, #00a8e8);
            padding: 2rem; border-radius: 16px; color: white;
            margin-bottom: 2rem; text-align: center;
        }
        .result-card {
            background: #e8f4f8; border: 2px solid #1a7abf; 
            border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── 7. Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state['name']}")
        authenticator.logout("Logout", "sidebar")
        st.markdown("---")
        st.metric("Symptoms Tracked", "132")
        st.metric("Diseases Covered", "41")
        st.info("Educational purposes only.")

    # ── 8. Main Dashboard ────────────────────────────────────────────────────
    st.markdown('<div class="main-header"><h1>🩺 MedAssist AI</h1></div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model files (model.pkl / symptoms.pkl) are missing from the repo!")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Symptom Checker", "📋 Disease Reference"])

    with tab1:
        st.subheader("Select Your Symptoms")
        
        # CATEGORIES (Shortened for brevity, add your full list as needed)
        symptom_categories = {
            "🌡️ General": ["fever", "fatigue", "chills", "weakness"],
            "🫁 Respiratory": ["cough", "shortness_of_breath", "sore_throat"],
            "🤢 Digestive": ["nausea", "vomiting", "diarrhea"]
        }

        selected_symptoms = []
        cols = st.columns(2)
        for i, (cat, syms) in enumerate(symptom_categories.items()):
            valid = [s for s in syms if s in SYMPTOMS]
            with cols[i % 2]:
                with st.expander(cat, expanded=True):
                    for s in valid:
                        if st.checkbox(s.replace("_", " ").title(), key=s):
                            selected_symptoms.append(s)

        if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
            if len(selected_symptoms) < 2:
                st.warning("⚠️ Select at least 2 symptoms.")
            else:
                # Prediction Logic
                input_v = [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
                pred = model.predict(np.array(input_v).reshape(1, -1))[0]
                
                st.markdown(f"""<div class="result-card">
                    <h2>Predicted: {pred.replace('_', ' ')}</h2>
                    <p>Severity: {SEVERITY_MAP.get(pred, '🟡 MODERATE')}</p>
                </div>""", unsafe_allow_html=True)
                
                st.success(f"**Recommended Medicine:** {MEDICINE_MAP.get(pred, 'Consult a physician.')}")

    with tab2:
        df_ref = pd.DataFrame([{"Disease": k, "Medicine": v} for k,v in MEDICINE_MAP.items()])
        st.dataframe(df_ref, use_container_width=True)
