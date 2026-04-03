import streamlit as st
import pickle
import numpy as np
import pandas as pd
import hashlib

# ── 1. Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="MedAssist AI", page_icon="🩺", layout="wide")

# ── 2. Simple Secure Authentication ──────────────────────────────────────────
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# Your Credentials
USER = "Arjunmedico"
# This is the SHA256 hash for 'HealthPrediction'
PASS_HASH = "85e33d06283b06387084534726e632d44f6f8749e7b45889724128f645851f5a"

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("## 🩺 MedAssist AI Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if username == USER and check_hashes(password, PASS_HASH):
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Incorrect Username or Password")
else:
    # ── 3. Authenticated App Content ─────────────────────────────────────────
    with st.sidebar:
        st.write(f"Logged in as: **{USER}**")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.markdown("---")
        st.metric("Model", "Random Forest")

    # ── Medicine Database & Model Loading (Your Original Logic) ──────────────
    @st.cache_resource
    def load_model():
        try:
            model = pickle.load(open("model.pkl", "rb"))
            symptoms = pickle.load(open("symptoms.pkl", "rb"))
            return model, symptoms
        except: return None, None

    model, SYMPTOMS = load_model()

    st.title("🩺 MedAssist AI Dashboard")
    
    tab1, tab2 = st.tabs(["🔍 Symptom Checker", "📋 Reference"])
    
    with tab1:
        st.subheader("Select Symptoms")
        # (Paste your checkbox and prediction logic here)
        st.info("Select symptoms and click predict.")
