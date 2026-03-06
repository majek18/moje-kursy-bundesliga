import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# --- 1. KONFIGURACJA ---
st.set_page_config(page_title="Analiza AI - Football Predictor", layout="wide")

# Konfiguracja Gemini (Gemini 3 Flash w Free Tier)
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # Najbardziej stabilna nazwa
    except Exception as e:
        st.error(f"Błąd konfiguracji AI: {e}")
        model = None
else:
    st.warning("⚠️ Brak klucza API w Secrets (GEMINI_API_KEY).")
    model = None

# --- 2. DANE ---
@st.cache_data
def get_data():
    return pd.DataFrame({
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 'Arsenal', 'Manchester City'],
        'H_GF': [4.0, 2.3, 2.1, 2.2, 2.3, 2.4],
        'H_GA': [1.0, 0.9, 0.9, 1.4, 0.6, 0.7],
        'A_GF': [3.3, 1.9, 1.7, 1.6, 1.6, 1.6],
        'A_GA': [0.9, 1.2, 1.5, 1.3, 0.8, 1.1]
    })

df = get_data()

# --- 3. INTERFEJS ---
st.title("⚽ Inteligentna Analiza Meczu")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

# Obliczenia Poisson (Statystyka)
avg_g = 1.5
h_stat = df[df['Team'] == h_team].iloc[0]
a_stat = df[df['Team'] == a_team].iloc[0]

l_h = (h_stat['H_GF'] / avg_g) * (a_stat['A_GA'] / avg_g) * avg_g
l_a = (a_stat['A_GF'] / avg_g) * (h_stat['H_GA'] / avg_g) * avg_g

# Wyświetlenie wyniku statystycznego
st.subheader(f"📊 Wynik z modelu matematycznego: {l_h:.2f} - {l_a:.2f}")

st.divider()

# --- 4. ANALIZA AI (PRZYCISK) ---
st.subheader("🤖 Ekspercka Analiza AI")
st.write("Kliknij poniżej, aby Gemini przeprowadziło głęboką analizę tego zestawienia.")

if st.button("🚀 Generuj Analizę AI", type="primary"):
    if model:
        with st.spinner("Sztuczna inteligencja analizuje dane..."):
            try:
                # Przygotowanie promptu dla AI
                prompt = f"""
                Działaj jako profesjonalny analityk sportowy. 
                Przeanalizuj mecz: {h_team} (Gospodarz) vs {a_team} (Gość).
                Statystyczne przewidywanie goli to: {h_team} {l_h:.2f} gola, {a_team} {l_a:.2f} gola.
                
                Napisz krótką (max 200 słów) analizę zawierającą:
                1. Kto ma przewagę taktyczną?
                2. Na co zwrócić uwagę (np. obrona gospodarzy vs atak gości)?
                3. Końcowy werdykt (kto wygra lub czy będzie remis).
                """
                
                # Generowanie treści
                response = model.generate_content(prompt)
                
                # Wyświetlenie wyniku w ładnej ramce
                st.info(response.text)
                
            except Exception as e:
                st.error(f"Błąd podczas generowania analizy: {e}")
    else:
        st.error("AI nie jest skonfigurowane. Sprawdź klucz API.")
