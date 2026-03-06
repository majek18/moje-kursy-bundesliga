import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Expert Model", layout="wide")

# --- KOMPLETNE DANE (Z uwzględnieniem Twoich kategorii) ---
@st.cache_data
def load_data():
    # Dane oparte na screenach + wartości szacunkowe dla formy i xG
    teams = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Koln', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        # 1. Dom (Home)
        'H_GF': [4.0, 2.3, 2.2, 1.7, 2.2, 2.1, 1.8, 1.9, 1.3, 1.4, 1.4, 1.1, 1.7, 1.1, 1.2, 1.2, 1.6, 1.1], # Gole strzelone dom
        'H_GA': [1.0, 0.9, 1.2, 1.0, 1.4, 0.9, 1.5, 1.1, 1.5, 1.4, 1.2, 1.7, 1.6, 1.2, 1.6, 1.7, 2.1, 2.2], # Gole stracone dom
        'H_xG': [3.4, 2.0, 2.1, 2.1, 2.6, 2.2, 1.7, 1.8, 1.3, 1.5, 1.6, 1.4, 1.5, 1.9, 1.0, 1.6, 1.5, 1.5], # xG dom
        'H_xA': [1.0, 1.2, 1.3, 1.3, 1.5, 0.9, 1.2, 1.1, 1.7, 1.3, 1.6, 1.7, 1.6, 1.5, 1.5, 1.4, 1.8, 2.0], # xA (xGA) dom
        # 2. Wyjazd (Away)
        'A_GF': [3.3, 1.9, 1.8, 2.2, 1.6, 1.7, 2.2, 1.0, 1.2, 1.0, 0.6, 1.1, 1.0, 1.2, 0.8, 0.9, 1.2, 0.7], 
        'A_GA': [0.9, 1.2, 1.4, 1.7, 1.3, 1.5, 2.6, 2.1, 2.0, 1.7, 1.7, 1.5, 1.8, 2.1, 1.7, 1.9, 2.2, 2.2],
        'A_xG': [2.7, 1.7, 1.6, 1.8, 1.7, 1.8, 1.4, 1.0, 1.2, 1.3, 1.0, 1.4, 1.4, 1.3, 0.9, 1.0, 1.3, 1.2],
        'A_xA': [1.2, 1.4, 1.9, 1.4, 1.3, 1.6, 2.0, 1.9, 2.1, 1.6, 1.9, 1.5, 2.1, 2.3, 2.1, 2.1, 2.1, 2.4],
        # 3. Forma (Ostatnie 6 meczów - ogółem)
        'F_GF': [3.1, 2.0, 1.9, 1.8, 2.0, 1.8, 1.6, 1.4, 1.3, 1.2, 1.0, 1.2, 1.3, 1.1, 0.9, 1.1, 1.2, 0.8],
        'F_GA': [0.9, 1.1, 1.5, 1.3, 1.4, 1.2, 1.9, 1.6, 1.7, 1.5, 1.5, 1.6, 1.9, 1.7, 1.6, 1.8, 2.0, 2.1]
    }
    return pd.DataFrame(teams)

df = load_data()

# --- ŚREDNIE LIGOWE (Stałe dla Bundesligi) ---
AVG_HOME_GF = df['H_GF'].mean() # Średnia goli gospodarza w lidze
AVG_AWAY_GF = df['A_GF'].mean() # Średnia goli gościa w lidze

# --- SIDEBAR: MODYFIKATORY ---
st.sidebar.header("🛠️ Modyfikatory Składu")
mod_h = st.sidebar.slider("Modyfikator Gospodarza (np. kontuzje)", 0.70, 1.10, 1.00, 0.05)
mod_a = st.sidebar.slider("Modyfikator Gościa (np. kontuzje)", 0.70, 1.10, 1.00, 0.05)

# --- WYBÓR MECZU ---
st.header("⚽ Kalkulator Prawdopodobieństwa (Metoda Autorska)")
c1, c2 = st.columns(2)
with c1: h_t = st.selectbox("Gospodarz (A)", df['Team'], index=0)
with c2: a_t = st.selectbox("Gość (B)", df['Team'], index=1)

# --- LOGIKA OBLICZEŃ (TWOJA METODA) ---
def get_poisson_params(home_name, away_name):
    h = df[df['Team'] == home_name].iloc[0]
    a = df[df['Team'] == away_name].iloc[0]
    
    # 1. Proporcje Drużyna A (Gospodarz)
    # Atak: 50% Dom + 30% Forma + 20% xG
    h_atk_val = (h['H_GF'] * 0.5) + (h['F_GF'] * 0.3) + (h['H_xG'] * 0.2)
    # Obrona: 50% Dom + 30% Forma + 20% xA (GA)
    h_def_val = (h['H_GA'] * 0.5) + (h['F_GA'] * 0.3) + (h['H_xA'] * 0.2)
    
    # 2. Proporcje Drużyna B (Gość)
    # Atak: 50% Wyjazd + 30% Forma + 20% xG
    a_atk_val = (a['A_GF'] * 0.5) + (a['F_GF'] * 0.3) + (a['A_xG'] * 0.2)
    # Obrona: 50% Wyjazd + 30% Forma + 20% xA
    a_def_val = (a['A_GA'] * 0.5) + (a['F_GA'] * 0.3) + (a['A_xA'] * 0.2)
    
    # 3. Współczynniki siły (dzielone przez średnią ligową)
    strength_h_atk = h_atk_val / AVG_HOME_GF
    strength_h_def = h_def_val / AVG_AWAY_GF
    strength_a_atk = a_atk_val / AVG_AWAY_GF
    strength_a_def = a_def_val / AVG_HOME_GF
    
    # 4. Finalna Lambda (Oczekiwane gole w tym meczu)
    # Atak A * Obrona B * Średnia Domowa * Modyfikator
    lamb = strength_h_atk * strength_a_def * AVG_HOME_GF * mod_h
    # Atak B * Obrona A * Średnia Wyjazdowa * Modyfikator
