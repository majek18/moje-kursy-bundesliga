import streamlit as st
import pandas as pd
import google.generativeai as genai

# Konfiguracja strony
st.set_page_config(page_title="Football AI Predictor", layout="wide")

# Pobieranie klucza z Secrets
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Używamy samej nazwy modelu, by uniknąć błędu v1beta
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Błąd konfiguracji AI: {e}")
        model = None
else:
    st.warning("⚠️ Brak klucza GEMINI_API_KEY w Secrets.")
    model = None

# Przykładowe dane
df = pd.DataFrame({'Team': ['Bayern Munich', 'Borussia Dortmund', 'Arsenal', 'Manchester City']})

st.title("⚽ Analiza AI Gemini")
h_team = st.selectbox("Gospodarz", df['Team'])
a_team = st.selectbox("Gość", df['Team'])

st.divider()

if st.button("🚀 Generuj Analizę AI", type="primary"):
    if model:
        with st.spinner("AI analizuje mecz..."):
            try:
                # Prosty prompt bez błędnych parametrów 'status'
                prompt = f"Przeanalizuj mecz: {h_team} vs {a_team}. Kto wygra?"
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"Błąd Gemini: {e}")
