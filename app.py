import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Matrix Pro", layout="wide")

# --- DANE ZACIĄGNIĘTE ZE SCREENÓW ---
@st.cache_data
def load_data():
    teams = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Koln', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        # Dane Domowe (Home)
        'H_M': [12, 12, 12, 12, 12, 12, 12, 11, 13, 12, 13, 12, 12, 12, 11, 12, 12, 12],
        'H_GF': [48, 28, 27, 21, 27, 25, 22, 21, 17, 17, 19, 14, 21, 13, 13, 14, 19, 13],
        'H_GA': [12, 11, 14, 12, 17, 11, 18, 12, 19, 17, 16, 21, 19, 14, 18, 21, 26, 27],
        'H_xG': [41.19, 23.95, 24.86, 25.33, 31.78, 27.12, 20.28, 20.43, 17.09, 18.15, 20.65, 17.57, 18.10, 23.06, 11.05, 19.14, 18.18, 17.69],
        'H_xGA': [12.53, 14.78, 15.40, 16.22, 18.10, 10.98, 15.09, 11.72, 21.74, 15.75, 20.60, 20.81, 19.77, 18.40, 16.95, 16.33, 22.03, 24.75],
        # Dane Wyjazdowe (Away)
        'A_M': [12, 12, 12, 12, 12, 12, 12, 13, 11, 12, 11, 12, 12, 12, 13, 12, 12, 12],
        'A_GF': [40, 23, 22, 27, 19, 20, 26, 13, 13, 12, 7, 13, 12, 14, 10, 11, 14, 9],
        'A_GA': [11, 14, 17, 20, 16, 18, 31, 27, 22, 21, 19, 18, 22, 25, 22, 23, 27, 26],
        'A_xG': [32.58, 20.39, 19.43, 21.60, 21.10, 21.28, 17.10, 13.73, 12.93, 15.97, 10.96, 16.81, 16.63, 16.06, 12.33, 12.53, 15.56, 14.96],
        'A_xGA': [14.51, 16.88, 22.87, 17.48, 16.02, 19.48, 23.57, 24.82, 23.37, 19.29, 20.78, 18.23, 25.55, 27.34, 27.00, 24.95, 24.95, 28.53]
    }
    return pd.DataFrame(teams)

df = load_data()

# --- BOCZNY PANEL: INTELIGENTNE SUWAKI ---
st.sidebar.header("⚖️ Konfiguracja Wag")
st.sidebar.info("Zmień jedną wagę, a pozostałe dostosują się automatycznie.")

# Inicjalizacja wag w sesji
if 'w_h_g' not in st.session_state:
    st.session_state.w_h_g, st.session_state.w_h_xg = 0.25, 0.25
    st.session_state.w_a_g, st.session_state.w_a_xg = 0.25, 0.25

def update_weights(key):
    total_others = 1.0 - st.session_state[key]
    current_others_sum = sum(st.session_state[k] for k in st.session_state if k != key and k.startswith('w_'))
    for k in st.session_state:
        if k != key and k.startswith('w_'):
            st.session_state[k] = (st.session_state[k] / current_others_sum) * total_others

w_h_g = st.sidebar.slider("Gole Dom (%)", 0.0, 1.0, st.session_state.w_h_g, key='w_h_g', on_change=update_weights, args=('w_h_g',))
w_h_xg = st.sidebar.slider("xG Dom (%)", 0.0, 1.0, st.session_state.w_h_xg, key='w_h_xg', on_change=update_weights, args=('w_h_xg',))
w_a_g = st.sidebar.slider("Gole Wyjazd (%)", 0.0, 1.0, st.session_state.w_a_g, key='w_a_g', on_change=update_weights, args=('w_a_g',))
w_a_xg = st.sidebar.slider("xG Wyjazd (%)", 0.0, 1.0, st.session_state.w_a_xg, key='w_a_xg', on_change=update_weights, args=('w_a_xg',))

# --- WYBÓR DRUŻYN ---
st.header("🏟️ Symulacja Meczu")
col_s1, col_s2 = st.columns(2)
with col_s1:
    h_team = st.selectbox("Wybierz Gospodarza", df['Team'], index=0)
with col_s2:
    a_team = st.selectbox("Wybierz Gościa", df['Team'], index=5)

# --- OBLICZENIA ---
def calculate_lambda_mu(home, away):
    h_stats = df[df['Team'] == home].iloc[0]
    a_stats = df[df['Team'] == away].iloc[0]
    
    # Atak Gospodarza: Średnia ważona (G_H i xG_H)
    home_attack = (h_stats['H_GF']/h_stats['H_M'] * w_h_g) + (h_stats['H_xG']/h_stats['H_M'] * w_h_xg)
    # Obrona Gościa: Średnia ważona (GA_A i xGA_A)
    away_defense = (a_stats['A_GA']/a_stats['A_M'] * w_a_g) + (a_stats['A_xGA']/a_stats['A_M'] * w_a_xg)
    
    # Atak Gościa: Średnia ważona (G_A i xG_A)
    away_attack = (a_stats['A_GF']/a_stats['A_M'] * w_a_g) + (a_stats['A_xG']/a_stats['A_M'] * w_a_xg)
    # Obrona Gospodarza: Średnia ważona (GA_H i xGA_H)
    home_defense = (h_stats['H_GA']/h_stats['H_M'] * w_h_g) + (h_stats['H_xGA']/h_stats['H_M'] * w_h_xg)
    
    # Przykładowe średnie ligowe z Twoich danych (ok. 1.5 gola na mecz)
    return home_attack * away_defense / 1.5, away_attack * home_defense / 1.5

lamb, mu = calculate_lambda_mu(h_team, a_team)

# --- MACIERZ WYNIKÓW ---
size = 6
matrix = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        matrix[i, j] = poisson.pmf(i, lamb) * poisson.pmf(j, mu)

# --- WIZUALIZACJA WYNIKÓW ---
p1 = np.sum(np.tril(matrix, -1))
px = np.sum(np.diag(matrix))
p2 = np.sum(np.triu(matrix, 1))

res1, resx, res2 = st.columns(3)
res1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
resx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
res2.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

st.write("### 📊 Macierz Prawdopodobieństwa Wyników")
st.write("Kolumny to gole Gościa, wiersze to gole Gospodarza. Ciemniejszy kolor = większa szansa.")

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False, 
            xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
plt.xlabel(f"Gole: {a_team}")
plt.ylabel(f"Gole: {h_team}")
st.pyplot(fig)

st.divider()
st.caption("Dane na podstawie dostarczonych statystyk sezonu 25/26. Model Poisson.")
