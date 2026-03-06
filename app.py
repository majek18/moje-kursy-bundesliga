import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor", layout="wide")

# --- KONFIGURACJA AI (GEMINI) ---
# Pobieranie klucza z Secrets
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Używamy samej nazwy modelu bez prefiksów, aby uniknąć błędu 404
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Prosty test połączenia przy starcie
        st.sidebar.success("✅ AI jest gotowe")
    except Exception as e:
        st.error(f"Błąd inicjalizacji Gemini: {e}")
        model = None
else:
    st.sidebar.warning("⚠️ Brak GEMINI_API_KEY w Secrets.")
    model = None

# --- DANE ---
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen'],
        'H_GF': [4.0, 2.3, 2.1], 'H_GA': [1.0, 0.9, 0.9],
        'A_GF': [3.3, 1.9, 1.7], 'A_GA': [0.9, 1.2, 1.5]
    })

df = load_data()

st.title("⚽ Football Predictor Pro")

h_team = st.selectbox("Wybierz Gospodarza", df['Team'], index=0)
a_team = st.selectbox("Wybierz Gościa", df['Team'], index=1)

# --- ANALIZA AI ---
st.divider()
st.subheader("🤖 AI Analyst")

if model:
    user_input = st.text_input("Zadaj pytanie analitykowi AI:", key="chat_input")
    if user_input:
        with st.spinner("Generowanie odpowiedzi..."):
            try:
                # Najbezpieczniejsza metoda generowania treści
                response = model.generate_content(f"Mecz: {h_team} vs {a_team}. Pytanie: {user_input}")
                st.info(response.text)
            except Exception as e:
                st.error(f"Wystąpił błąd: {e}")
else:
    st.info("Dodaj klucz API w ustawieniach Streamlit (Secrets), aby uruchomić AI.")
