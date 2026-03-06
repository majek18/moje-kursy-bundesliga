import streamlit as st
import pandas as pd
import google.generativeai as genai

# Konfiguracja strony
st.set_page_config(page_title="Football AI Analysis", layout="wide")

# Konfiguracja Gemini (Free Tier)
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Używamy pełnej ścieżki modelu, aby naprawić błąd 404
        model = genai.GenerativeModel('models/gemini-1.5-flash')
    except Exception as e:
        st.error(f"Błąd konfiguracji: {e}")
        model = None
else:
    st.error("⚠️ Brak klucza GEMINI_API_KEY w Secrets.")
    model = None

# Przykładowe dane
df = pd.DataFrame({'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'Arsenal', 'Manchester City']})

st.title("⚽ Analiza AI Gemini")
h_team = st.selectbox("Gospodarz", df['Team'], index=0)
a_team = st.selectbox("Gość", df['Team'], index=1)

st.divider()

# ANALIZA JEDNYM PRZYCISKIEM
if st.button("🚀 Generuj Analizę AI", type="primary"):
    if model:
        with st.spinner("Sztuczna inteligencja analizuje mecz..."):
            try:
                # Najprostszy prompt, by uniknąć błędów formatowania
                prompt_text = f"Przeanalizuj mecz piłki nożnej: {h_team} vs {a_team}. Kto wygra?"
                response = model.generate_content(prompt_text)
                st.info(response.text)
            except Exception as e:
                # Jeśli tu wystąpi 404, upewnij się, że masz bibliotekę >= 0.8.3 w requirements.txt
                st.error(f"Błąd Gemini: {e}")
    else:
        st.warning("Model AI nie jest gotowy.")
