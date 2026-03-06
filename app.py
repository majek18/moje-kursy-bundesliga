import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# --- 1. KONFIGURACJA ---
st.set_page_config(page_title="Football Predictor GPT", layout="wide")

# Pobranie klucza z Secrets
# Upewnij się, że w Settings -> Secrets masz: OPENAI_API_KEY = "sk-..."
api_key = st.secrets.get("OPENAI_API_KEY")

if api_key:
    try:
        client = OpenAI(api_key=api_key)
        st.sidebar.success("✅ ChatGPT połączony")
    except Exception as e:
        st.error(f"Błąd inicjalizacji OpenAI: {e}")
        client = None
else:
    st.sidebar.warning("⚠️ Brak klucza OPENAI_API_KEY w Secrets.")
    client = None

# --- 2. DANE ---
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'Arsenal', 'Manchester City'],
        'H_GF': [4.0, 2.3, 2.1, 2.3, 2.4],
        'H_GA': [1.0, 0.9, 0.9, 0.6, 0.7],
        'A_GF': [3.3, 1.9, 1.7, 1.6, 1.6],
        'A_GA': [0.9, 1.2, 1.5, 0.8, 1.1]
    })

df = load_data()

# --- 3. INTERFEJS ---
st.title("⚽ Football Predictor & GPT Analysis")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

# Obliczenia statystyczne (Poisson)
avg_g = 1.5
h_stat = df[df['Team'] == h_team].iloc[0]
a_stat = df[df['Team'] == a_team].iloc[0]

l_h = (h_stat['H_GF'] / avg_g) * (a_stat['A_GA'] / avg_g) * avg_g
l_a = (a_stat['A_GF'] / avg_g) * (h_stat['H_GA'] / avg_g) * avg_g

st.subheader(f"📊 Przewidywany wynik: {l_h:.2f} - {l_a:.2f}")

st.divider()

# --- 4. ANALIZA GPT (PRZYCISK) ---
st.subheader("🤖 Analiza ekspercka GPT")

if st.button("🚀 Generuj Analizę GPT", type="primary"):
    if client:
        with st.spinner("ChatGPT analizuje mecz..."):
            try:
                # Wywołanie modelu gpt-4o-mini (szybki i tani)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Jesteś ekspertem piłkarskim."},
                        {"role": "user", "content": f"Analizuj mecz: {h_team} vs {a_team}. Statystyki: {l_h:.2f} gola gospodarzy, {l_a:.2f} gola gości. Kto wygra?"}
                    ]
                )
                # Wyświetlenie wyniku
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Błąd OpenAI: {e}")
    else:
        st.error("Skonfiguruj klucz API OpenAI w ustawieniach.")
