import streamlit as st
import pandas as pd
import google.generativeai as genai

# Konfiguracja strony
st.set_page_config(page_title="Analiza AI", layout="wide")

# Konfiguracja AI (Gemini 3 Flash)
api_key = st.secrets.get("GEMINI_API_KEY")

if api_key:
    try:
        genai.configure(api_key=api_key)
        # UWAGA: Używamy samej nazwy 'gemini-1.5-flash', biblioteka sama dobierze API
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Błąd inicjalizacji: {e}")
        model = None
else:
    st.error("Brak klucza API w Secrets.")
    model = None

# Przykładowe dane
df = pd.DataFrame({'Team': ['Bayern Munich', 'Borussia Dortmund', 'Arsenal', 'Real Madrid']})

st.title("⚽ Ekspercka Analiza AI")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

st.divider()

# PRZYCISK ANALIZY - to rozwiązuje problemy z czatem
if st.button("🚀 Generuj Analizę AI", type="primary"):
    if model:
        with st.spinner("Gemini analizuje mecz..."):
            try:
                # Prosty prompt bez zbędnych konfiguracji (GenerationConfig), które wywalały błędy
                prompt = f"Przeanalizuj mecz piłkarski: {h_team} vs {a_team}. Kto ma większe szanse na wygraną i dlaczego?"
                response = model.generate_content(prompt)
                
                # Wyświetlenie wyniku
                st.info(response.text)
            except Exception as e:
                # Jeśli tu wystąpi 404, spróbuj zmienić w kodzie wyżej na 'gemini-pro'
                st.error(f"Błąd Gemini: {e}")
    else:
        st.warning("AI nie jest gotowe. Sprawdź klucz API.")
