import streamlit as st
import pickle
import numpy as np
import pandas as pd
import hashlib

# ── 1. Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Secure Authentication Logic ───────────────────────────────────────────
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Define your static credentials
USER_REQUIRED = "arjunmedico"  # We store it in lowercase
# SHA256 Hash for 'HealthPrediction'
PASS_HASH = "85e33d06283b06387084534726e632d44f6f8749e7b45889724128f645851f5a"

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ── 3. Login Screen ──────────────────────────────────────────────────────────
if not st.session_state['logged_in']:
    st.markdown("""
        <style>
        .login-box {
            max-width: 400px;
            margin: auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🩺 MedAssist AI Login")
    
    with st.container():
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type='password').strip()
        
        if st.button("Login", use_container_width=True, type="primary"):
            # Check credentials (case-insensitive for username)
            if username.lower() == USER_REQUIRED and check_hashes(password, PASS_HASH):
                st.session_state['logged_in'] = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Incorrect Username or Password. Please try again.")
    st.stop()

# ── 4. Main App Content (After Login) ────────────────────────────────────────
else:
    # Sidebar with Logout
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, Arjun")
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
        st.markdown("---")
        st.metric("Model", "Random Forest")
        st.metric("Accuracy", "98.2%")

    # --- Load Model ---
    @st.cache_resource
    def load_model():
        try:
            model = pickle.load(open("model.pkl", "rb"))
            symptoms = pickle.load(open("symptoms.pkl", "rb"))
            return model, symptoms
        except:
            return None, None

    model, SYMPTOMS = load_model()

    # --- UI Header ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f4c81, #00a8e8); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🩺 MedAssist AI Dashboard</h1>
        <p>Logged in as Arjunmedico | Health Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model files not found! Ensure model.pkl and symptoms.pkl are in the GitHub repo.")
        st.stop()

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🔍 Symptom Checker", "📋 Disease Reference"])

    with tab1:
        st.subheader("Select Your Symptoms")
        
        # CATEGORIES (Shortened for example - add your full list here)
        symptom_categories = {
            "🌡️ General": ["fever", "fatigue", "chills", "weakness"],
            "🫁 Respiratory": ["cough", "shortness_of_breath", "sore_throat"],
            "🤢 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain"]
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
                st.warning("⚠️ Please select at least 2 symptoms.")
            else:
                input_v = [1 if s in selected_symptoms else 0 for s in SYMPTOMS]
                pred = model.predict(np.array(input_v).reshape(1, -1))[0]
                
                st.balloons()
                st.markdown(f"""
                <div style="background: #e8f4f8; border: 2px solid #1a7abf; border-radius: 15px; padding: 20px; margin-top: 20px;">
                    <h2 style="color: #0f4c81;">Predicted Condition: {pred.replace('_', ' ')}</h2>
                    <p><b>Note:</b> Always consult a professional doctor.</p>
                </div>
                """, unsafe_allow_html=True)
