import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

# Konfiguracja strony
st.set_page_config(page_title="Football Predictor", layout="wide")

# AI Setup - Naprawa błędu klucza i modelu 404
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Używamy stabilnej nazwy modelu
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Problem z konfiguracją AI: {e}")
        model = None
else:
    st.warning("⚠️ Brak klucza API w Secrets (GEMINI_API_KEY). Funkcje AI są wyłączone.")
    model = None

# Funkcja do danych
@st.cache_data
def get_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig'],
        'H_GF': [4.00, 2.33, 2.08, 2.25],
        'H_GA': [1.00, 0.92, 0.92, 1.42],
        'A_GF': [3.33, 1.92, 1.67, 1.58],
        'A_GA': [0.92, 1.17, 1.50, 1.33],
        'Logo_ID': [27, 16, 15, 23826]
    }
    return pd.DataFrame(data)

df = get_data()

st.title("⚽ Football Predictor Pro")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

# Prosta logika Poisson
avg_g = 1.5
h_stat = df[df['Team'] == h_team].iloc[0]
a_stat = df[df['Team'] == a_team].iloc[0]

l_h = (h_stat['H_GF'] / avg_g) * (a_stat['A_GA'] / avg_g) * avg_g
l_a = (a_stat['A_GF'] / avg_g) * (h_stat['H_GA'] / avg_g) * avg_g

st.metric("Przewidywane gole", f"{h_team} {l_h:.2f} - {l_a:.2f} {a_team}")

# Sekcja AI Analyst
st.divider()
st.subheader("🤖 AI Analyst")

if model:
    user_query = st.text_input("Zadaj pytanie analitykowi AI:", placeholder="Kto ma większą szansę na wygraną?")
    if user_query:
        with st.spinner("AI analizuje dane..."):
            prompt = f"Analizuj mecz: {h_team} vs {a_team}. Statystyki goli: {h_team}({l_h:.2f}), {a_team}({l_a:.2f}). Pytanie: {user_query}"
            try:
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"Błąd generowania odpowiedzi: {e}")
else:
    st.info("Aby odblokować AI, dodaj GEMINI_API_KEY w ustawieniach Streamlit Cloud.")
