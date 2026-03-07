import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import time

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor & AI Chat", layout="wide", page_icon="⚽")

# --- KONFIGURACJA GEMINI API ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("❌ Brakuje GOOGLE_API_KEY w Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-2.0-flash")

model = load_model()

# =================================================================
# --- DANE BAZOWE: BUNDESLIGA ---
# =================================================================

@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich','Borussia Dortmund','Hoffenheim','VfB Stuttgart','RB Leipzig','Bayer Leverkusen',
        'Eintracht Frankfurt','Freiburg','Augsburg','Union Berlin','Hamburger SV','Borussia M.Gladbach',
        'FC Cologne','Mainz 05','St. Pauli','Werder Bremen','Wolfsburg','FC Heidenheim'],

        'H_GF':[4.00,2.33,2.25,1.75,2.25,2.08,1.83,1.91,1.31,1.42,1.46,1.17,1.75,1.08,1.18,1.17,1.58,1.08],
        'H_GA':[1.00,0.92,1.17,1.00,1.42,0.92,1.50,1.09,1.46,1.42,1.23,1.75,1.58,1.17,1.64,1.75,2.17,2.25],
        'T_GF':[3.67,2.13,2.04,2.00,1.92,1.88,2.00,1.42,1.25,1.21,1.08,1.13,1.38,1.13,0.96,1.04,1.38,0.92],
        'T_GA':[0.96,1.04,1.29,1.33,1.38,1.21,2.04,1.63,1.71,1.58,1.46,1.63,1.71,1.63,1.67,1.83,2.21,2.21],

        'HxG_F':[3.43,2.00,2.07,2.11,2.65,2.26,1.69,1.86,1.31,1.51,1.59,1.46,1.51,1.92,1.00,1.60,1.52,1.47],
        'HxG_A':[1.04,1.23,1.28,1.35,1.51,0.92,1.26,1.07,1.67,1.31,1.58,1.73,1.65,1.53,1.54,1.36,1.84,2.06],

        'TxG_F':[3.07,1.85,1.85,1.96,2.20,2.02,1.56,1.42,1.25,1.42,1.32,1.43,1.45,1.63,0.97,1.32,1.41,1.36],
        'TxG_A':[1.13,1.32,1.59,1.40,1.42,1.27,1.61,1.52,1.88,1.46,1.72,1.63,1.89,1.90,1.83,1.72,1.96,2.22],

        'A_GF':[3.33,1.92,1.83,2.25,1.58,1.67,2.17,1.00,1.18,1.00,0.64,1.08,1.00,1.17,0.77,0.92,1.17,0.75],
        'A_GA':[0.92,1.17,1.42,1.67,1.33,1.50,2.58,2.08,2.00,1.75,1.73,1.50,1.83,2.08,1.69,1.92,2.25,2.17],

        'AxG_F':[2.72,1.70,1.62,1.80,1.76,1.77,1.43,1.06,1.18,1.06,1.00,1.40,1.39,1.34,0.95,1.04,1.30,1.25],
        'AxG_A':[1.21,1.41,1.91,1.46,1.34,1.62,1.96,1.91,2.12,1.61,1.89,1.52,2.13,2.28,2.08,2.08,2.08,2.38],

        'Logo_ID':[27,16,533,79,23826,15,24,60,167,89,41,18,3,39,35,86,82,2036]
    }
    return pd.DataFrame(data)

# =================================================================
# --- PREMIER LEAGUE ---
# =================================================================

@st.cache_data
def load_premier_league():
    data = {
        'Team':['Arsenal','Manchester City','Manchester United','Aston Villa','Chelsea','Liverpool','Brentford',
        'Everton','Bournemouth','Fulham','Sunderland','Newcastle','Crystal Palace','Brighton','Leeds',
        'Tottenham','Nottingham Forest','West Ham','Burnley','Wolves'],

        'H_GF':[2.35,2.40,1.92,1.40,1.64,1.85,1.71,1.20,1.40,1.60,1.57,1.86,1.00,1.46,1.46,1.20,0.92,1.21,1.07,1.06],
        'H_GA':[0.64,0.73,1.14,1.00,1.14,1.14,1.07,1.26,1.00,1.20,0.92,1.60,1.28,1.06,1.33,1.66,1.35,1.92,1.64,1.93],

        'T_GF':[1.96,2.03,1.75,1.34,1.82,1.65,1.51,1.17,1.51,1.37,1.03,1.44,1.13,1.31,1.27,1.34,0.96,1.20,1.10,0.73],
        'T_GA':[0.73,0.93,1.37,1.17,1.17,1.34,1.37,1.13,1.58,1.48,1.13,1.48,1.20,1.24,1.65,1.58,1.48,1.86,2.00,1.73],

        'HxG_F':[2.05,2.23,2.13,1.36,2.14,1.90,2.07,1.36,1.63,1.39,1.17,2.19,1.94,1.41,1.76,1.24,1.54,1.39,1.03,1.14],
        'HxG_A':[0.74,1.07,1.01,1.32,1.54,1.06,1.31,1.44,0.75,1.35,1.46,1.45,1.51,1.31,1.32,1.58,1.59,1.66,1.88,1.73],

        'TxG_F':[1.96,2.01,1.91,1.34,2.12,1.86,1.76,1.30,1.71,1.26,1.00,1.63,1.67,1.45,1.51,1.18,1.20,1.29,0.94,0.93],
        'TxG_A':[0.79,1.19,1.27,1.54,1.47,1.27,1.47,1.51,1.45,1.58,1.61,1.37,1.50,1.47,1.54,1.55,1.72,1.84,2.16,1.74],

        'A_GF':[1.62,1.64,1.60,1.28,2.00,1.46,1.33,1.14,1.64,1.14,0.53,1.00,1.26,1.14,1.00,1.50,1.00,1.20,1.13,0.35],
        'A_GA':[0.81,1.14,1.60,1.35,1.20,1.53,1.66,1.00,2.21,1.78,1.40,1.35,1.13,1.42,2.00,1.50,1.83,2.08,2.33,1.50],

        'AxG_F':[1.87,1.78,1.70,1.32,2.10,1.81,1.48,1.22,1.79,1.11,0.91,1.03,1.43,1.48,1.23,1.10,0.90,1.20,0.85,0.68],
        'AxG_A':[0.84,1.31,1.51,1.78,1.41,1.47,1.62,1.59,2.20,1.83,1.75,1.27,1.49,1.64,1.77,1.53,1.85,2.04,2.43,1.75],

        'Logo_ID':[11,281,985,405,631,31,1148,29,1003,931,289,762,873,1237,399,148,703,379,1132,543]
    }
    return pd.DataFrame(data)

# =================================================================
# --- TUTAJ ZOSTAJE CAŁY TWÓJ KOD ANALITYCZNY ---
# (poisson, macierz, symulacje itd.)
# =================================================================

# (Twoja funkcja render_league_ui pozostaje dokładnie taka sama jak wysłałeś)

# Wywołanie
tab_bl, tab_pl = st.tabs(["🇩🇪 Bundesliga", "🏴 Premier League"])

with tab_bl:
    render_league_ui(load_bundesliga(), "Bundesliga")

with tab_pl:
    render_league_ui(load_premier_league(), "Premier League")

# =================================================================
# ====================== CHATBOT NA DOLE ==========================
# =================================================================

st.markdown("---")
st.markdown("# 💬 Chatbot AI (Gemini)")

# pamięć rozmowy
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_call" not in st.session_state:
    st.session_state.last_call = 0

# historia
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
if prompt := st.chat_input("Zapytaj o piłkę, statystyki lub analizy..."):

    st.session_state.messages.append({"role": "user","content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        placeholder.markdown("*(AI analizuje...)*")

        try:

            # cooldown
            now = time.time()
            if now - st.session_state.last_call < 3:
                placeholder.warning("⏳ Poczekaj 3 sekundy między pytaniami.")
                st.stop()

            response = st.session_state.chat.send_message(prompt)

            st.session_state.last_call = time.time()

            if response.text:
                placeholder.markdown(response.text)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":response.text
                })

        except Exception as e:

            error = str(e)

            if "404" in error:
                st.error("❌ Model Gemini nie istnieje.")

            elif "429" in error:
                st.error("⏳ Limit zapytań API został przekroczony.")

            else:
                st.error(f"❌ Błąd AI: {error}")
