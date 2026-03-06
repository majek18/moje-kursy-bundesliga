import streamlit as st
import pandas as pd
from google import genai

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor 2.0", layout="wide")

# Pobieranie klucza z Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if api_key:
    try:
        # Inicjalizacja nowego klienta Google GenAI
        client = genai.Client(api_key=api_key)
        st.sidebar.success("✅ Gemini 2.0 Flash gotowy")
    except Exception as e:
        st.error(f"Błąd połączenia: {e}")
        client = None
else:
    st.error("⚠️ Brak klucza GEMINI_API_KEY w Secrets.")
    client = None

# --- DANE ---
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

# --- INTERFEJS ---
st.title("⚽ Football Analysis Gemini 2.0")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

st.divider()

# --- ANALIZA AI ---
if st.button("🚀 Generuj Analizę Gemini 2.0", type="primary"):
    if client:
        with st.spinner("Nowy model Gemini 2.0 analizuje dane..."):
            try:
                # Wywołanie modelu gemini-2.0-flash
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"Jesteś ekspertem. Przeanalizuj mecz {h_team} vs {a_team}. Kto wygra i dlaczego?"
                )
                # Wyświetlenie tekstu odpowiedzi
                st.info(response.text)
            except Exception as e:
                # To wyłapie błędy wersji, jeśli biblioteka w requirements będzie zła
                st.error(f"Błąd modelu: {e}")
    else:
        st.warning("Skonfiguruj klucz API.")
