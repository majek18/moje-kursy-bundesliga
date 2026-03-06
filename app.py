import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# Konfiguracja strony
st.set_page_config(page_title="Football Predictor", layout="wide")

# AI Setup - Naprawa błędu 404 i v1beta
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Próbujemy najbardziej bezpośredniej nazwy modelu
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={"status": "active"}
        )
        # Test połączenia (opcjonalny)
        st.sidebar.success("✅ Gemini skonfigurowane")
    except Exception as e:
        st.error(f"Błąd inicjalizacji: {e}")
        model = None
else:
    st.error("❌ Brak GEMINI_API_KEY w Secrets!")
    model = None

# Dane
@st.cache_data
def get_data():
    return pd.DataFrame({
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen'],
        'H_GF': [4.0, 2.3, 2.1], 'H_GA': [1.0, 0.9, 0.9],
        'A_GF': [3.3, 1.9, 1.7], 'A_GA': [0.9, 1.2, 1.5]
    })

df = get_data()
h_team = st.selectbox("Gospodarz", df['Team'], index=0)
a_team = st.selectbox("Gość", df['Team'], index=1)

st.divider()
st.subheader("🤖 AI Analyst")

if model:
    prompt = st.text_input("Zadaj pytanie:")
    if prompt:
        with st.spinner("Generowanie..."):
            try:
                # Wymuszamy najnowszą metodę generowania
                response = model.generate_content(f"Mecz: {h_team} vs {a_team}. {prompt}")
                st.write(response.text)
            except Exception as e:
                # Jeśli nadal 404, spróbujmy wyświetlić co widzi biblioteka
                st.error(f"Błąd modelu: {e}")
                if "404" in str(e):
                    st.info("Spróbuj zmienić nazwę modelu w kodzie na 'gemini-1.5-flash-latest'")
