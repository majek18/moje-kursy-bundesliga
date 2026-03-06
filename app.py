import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import google.generativeai as genai

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor & Copilot", layout="wide")

# --- KONFIGURACJA AI (GEMINI 3 FLASH) ---
# Pobieranie klucza z bezpiecznych ustawień Streamlit
gemini_key = st.secrets.get("GEMINI_API_KEY")

if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        # Używamy najbardziej stabilnej nazwy modelu
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.sidebar.success("✅ Copilot AI jest aktywny")
    except Exception as e:
        st.sidebar.error(f"Błąd inicjalizacji AI: {e}")
        model = None
else:
    st.sidebar.warning("⚠️ Brak klucza API w Secrets (GEMINI_API_KEY)")
    model = None

# --- DANE BAZOWE ---
@st.cache_data
def load_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 'St. Pauli', 'Arsenal', 'Manchester City', 'Liverpool'],
        'H_GF': [4.00, 2.33, 2.08, 2.25, 1.18, 2.35, 2.40, 1.85],
        'H_GA': [1.00, 0.92, 0.92, 1.42, 1.64, 0.64, 0.73, 1.14],
        'A_GF': [3.33, 1.92, 1.67, 1.58, 0.77, 1.62, 1.64, 1.46],
        'A_GA': [0.92, 1.17, 1.50, 1.33, 1.69, 0.81, 1.14, 1.53]
    }
    return pd.DataFrame(data)

df = load_data()

# --- INTERFEJS WYBORU MECZU ---
st.title("⚽ Football Predictor Pro")
col1, col2 = st.columns(2)

with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_stat = df[df['Team'] == h_team].iloc[0]

with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)
    a_stat = df[df['Team'] == a_team].iloc[0]

# --- PROSTA LOGIKA POISSONA ---
avg_league_g = 1.5
lambda_h = (h_stat['H_GF'] / avg_league_g) * (a_stat['A_GA'] / avg_league_g) * avg_league_g
lambda_a = (a_stat['A_GF'] / avg_league_g) * (h_stat['H_GA'] / avg_league_g) * avg_league_g

st.divider()
c1, c2 = st.columns(2)
c1.metric(f"Oczekiwane gole {h_team}", f"{lambda_h:.2f}")
c2.metric(f"Oczekiwane gole {a_team}", f"{lambda_a:.2f}")

# --- SEKCOJA CZATU: TWOJEGO COPILOTA ---
st.divider()
st.subheader("🤖 Twój Piłkarski Copilot (Gemini AI)")

# Inicjalizacja historii wiadomości
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obsługa wejścia użytkownika (Pasek na dole strony)
if prompt := st.chat_input("Zapytaj Copilota o ten mecz..."):
    # Dodanie wiadomości użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generowanie odpowiedzi AI
    if model:
        with st.chat_message("assistant"):
            try:
                # Tworzenie kontekstu dla Gemini
                context = f"""
                Jesteś profesjonalnym analitykiem sportowym (Copilot). 
                Analizujesz mecz: {h_team} vs {a_team}.
                Statystyki przewidywanych goli (Poisson): {h_team} ({lambda_h:.2f}), {a_team} ({lambda_a:.2f}).
                Odpowiedz na pytanie użytkownika rzeczowo i ekspercko: {prompt}
                """
                response = model.generate_content(context)
                answer = response.text
                st.markdown(answer)
                # Zapisanie odpowiedzi w historii sesji
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Błąd Copilota: {e}")
    else:
        st.error("Copilot nie jest gotowy. Sprawdź klucz API.")
