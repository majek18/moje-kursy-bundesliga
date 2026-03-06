import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from openai import OpenAI
import os

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor Pro", layout="wide", page_icon="⚽")

# --- STYLE CSS (Dla lepszego wyglądu) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- KONFIGURACJA AI (Hugging Face Router) ---
# Klucz pobierany z Settings -> Secrets w Streamlit Cloud
hf_token = st.secrets.get("HF_TOKEN")

def get_ai_client():
    if not hf_token:
        return None
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

client = get_ai_client()

# --- DANE BAZOWE: BUNDESLIGA ---
@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        'H_GF': [4.00, 2.33, 2.25, 1.75, 2.25, 2.08, 1.83, 1.91, 1.31, 1.42, 1.46, 1.17, 1.75, 1.08, 1.18, 1.17, 1.58, 1.08],
        'H_GA': [1.00, 0.92, 1.17, 1.00, 1.42, 0.92, 1.50, 1.09, 1.46, 1.42, 1.23, 1.75, 1.58, 1.17, 1.64, 1.75, 2.17, 2.25],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

df = load_bundesliga()

# --- INTERFEJS WYBORU MECZU ---
st.title("⚽ Football Predictor Pro")
st.markdown("Analityczny system przewidywania wyników oparty na rozkładzie Poissona i Llama 3.1.")

col_selection_1, col_selection_2 = st.columns(2)

with col_selection_1:
    h_team = st.selectbox("Wybierz Gospodarza", df['Team'].sort_values(), index=0)
    h_data = df[df['Team'] == h_team].iloc[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_data['Logo_ID']}.png", width=120)

with col_selection_2:
    a_team = st.selectbox("Wybierz Gościa", df['Team'].sort_values(), index=1)
    a_data = df[df['Team'] == a_team].iloc[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_data['Logo_ID']}.png", width=120)

# --- OBLICZENIA MATEMATYCZNE ---
avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

# Wyliczenie oczekiwanych goli (Expected Goals - lambda)
lambda_h = (h_data['H_GF'] / avg_h_gf) * (a_data['A_GA'] / avg_h_gf) * avg_h_gf
mu_a = (a_data['A_GF'] / avg_a_gf) * (h_data['H_GA'] / avg_a_gf) * avg_a_gf

# Obliczanie prawdopodobieństwa (Macierz wyników)
max_goals = 10
prob_matrix = np.outer(poisson.pmf(range(max_goals), lambda_h), poisson.pmf(range(max_goals), mu_a))

prob_h_win = np.sum(np.tril(prob_matrix, -1))
prob_draw = np.sum(np.diag(prob_matrix))
prob_a_win = np.sum(np.triu(prob_matrix, 1))

# --- WYNIKI ANALIZY ---
st.divider()
st.subheader("📊 Prawdopodobieństwo wyniku")
c1, c2, c3 = st.columns(3)
c1.metric(f"Zwycięstwo {h_team}", f"{prob_h_win:.1%}")
c2.metric("Remis", f"{prob_draw:.1%}")
c3.metric(f"Zwycięstwo {a_team}", f"{prob_a_win:.1%}")

st.info(f"Przewidywana liczba goli: **{h_team} {lambda_h:.2f}** - **{mu_a:.2f} {a_team}**")

# --- SEKCOJA CZATU AI ---
st.divider()
st.subheader("🤖 Ekspert AI (Llama 3.1)")

if client:
    # Inicjalizacja historii wiadomości
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Wyświetlanie historii
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Obsługa nowego pytania
    if prompt := st.chat_input("Zapytaj AI o analizę tego meczu lub typy bukmacherskie..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Przygotowanie kontekstu dla modelu
                context = f"""Jesteś profesjonalnym analitykiem sportowym. 
                Mecz: {h_team} vs {a_team}. 
                Statystyki matematyczne: 
                - Szansa na wygraną {h_team}: {prob_h_win:.1%}
                - Szansa na remis: {prob_draw:.1%}
                - Szansa na wygraną {a_team}: {prob_a_win:.1%}
                - Oczekiwane gole (xG): {h_team} ({lambda_h:.2f}), {a_team} ({mu_a:.2f}).
                
                Odpowiedz na pytanie użytkownika: {prompt}"""
                
                # Wywołanie modelu
                completion = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct:novita",
                    messages=[{"role": "user", "content": context}],
                    max_tokens=400,
                    temperature=0.7
                )
                
                response_text = completion.choices[0].message.content
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas generowania odpowiedzi: {e}")
else:
    st.warning("⚠️ Brak klucza HF_TOKEN w ustawieniach (Secrets). Dodaj go, aby odblokować analizę AI.")

# --- STOPKA ---
st.divider()
st.caption("Dane statystyczne oparte na historycznych wynikach Bundesligi. Pamiętaj, że sport jest nieprzewidywalny.")
