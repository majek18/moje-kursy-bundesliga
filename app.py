import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Predictor", layout="wide")

# --- DANE BUNDESLIGI (Z Twoich screenów) ---
@st.cache_data
def load_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        # Gole strzelone/stracone - Dom
        'H_GF': [4.00, 2.33, 2.25, 1.75, 2.25, 2.08, 1.83, 1.91, 1.31, 1.42, 1.46, 1.17, 1.75, 1.08, 1.18, 1.17, 1.58, 1.08],
        'H_GA': [1.00, 0.92, 1.17, 1.00, 1.42, 0.92, 1.50, 1.09, 1.46, 1.42, 1.23, 1.75, 1.58, 1.17, 1.64, 1.75, 2.17, 2.25],
        # Gole strzelone/stracone - Cały sezon
        'T_GF': [3.67, 2.13, 2.04, 2.00, 1.92, 1.88, 2.00, 1.42, 1.25, 1.21, 1.08, 1.13, 1.38, 1.13, 0.96, 1.04, 1.38, 0.92],
        'T_GA': [0.96, 1.04, 1.29, 1.33, 1.38, 1.21, 2.04, 1.63, 1.71, 1.58, 1.46, 1.63, 1.71, 1.63, 1.67, 1.83, 2.21, 2.21],
        # xG strzelone/stracone - Dom
        'HxG_F': [3.43, 2.00, 2.07, 2.11, 2.65, 2.26, 1.69, 1.86, 1.31, 1.51, 1.59, 1.46, 1.51, 1.92, 1.00, 1.60, 1.52, 1.47],
        'HxG_A': [1.04, 1.23, 1.28, 1.35, 1.51, 0.92, 1.26, 1.07, 1.67, 1.31, 1.58, 1.73, 1.65, 1.53, 1.54, 1.36, 1.84, 2.06],
        # xG strzelone/stracone - Cały sezon
        'TxG_F': [3.07, 1.85, 1.85, 1.96, 2.20, 2.02, 1.56, 1.42, 1.25, 1.42, 1.32, 1.43, 1.45, 1.63, 0.97, 1.32, 1.41, 1.36],
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
        # Gole/xG Wyjazd (dla Gościa)
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.33, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38]
    }
    return pd.DataFrame(data)

df = load_data()

# --- ŚREDNIE LIGOWE (Normalizacja) ---
avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

# --- SIDEBAR: KONFIGURACJA WAG (Twoje 50/30/20 po korekcie na brak formy) ---
st.sidebar.header("⚙️ Ustawienia Wag")
st.sidebar.info("Suma musi wynosić 100%. Dostosuj proporcje statystyk.")

if 'w' not in st.session_state:
    st.session_state.w = {'dom': 0.40, 'sezon': 0.30, 'xg_dom': 0.20, 'xg_sezon': 0.10}

def balance_weights(key):
    total_others = 1.0 - st.session_state[key]
    others = [k for k in st.session_state.w if k != key]
    current_others_sum = sum(st.session_state.w[k] for k in others)
    for k in others:
        st.session_state.w[k] = (st.session_state.w[k] / current_others_sum) * total_others
    st.session_state.w[key] = st.session_state[key]

w_dom = st.sidebar.slider("Gole Dom/Wyjazd (%)", 0.0, 1.0, st.session_state.w['dom'], key='dom', on_change=balance_weights, args=('dom',))
w_sezon = st.sidebar.slider("Gole Cały Sezon (%)", 0.0, 1.0, st.session_state.w['sezon'], key='sezon', on_change=balance_weights, args=('sezon',))
w_xg_dom = st.sidebar.slider("xG Dom/Wyjazd (%)", 0.0, 1.0, st.session_state.w['xg_dom'], key='xg_dom', on_change=balance_weights, args=('xg_dom',))
w_xg_sezon = st.sidebar.slider("xG Cały Sezon (%)", 0.0, 1.0, st.session_state.w['xg_sezon'], key='xg_sezon', on_change=balance_weights, args=('xg_sezon',))

# --- WYBÓR DRUŻYN ---
st.title("⚽ Bundesliga Match Predictor")
c1, c2 = st.columns(2)
with c1: h_team = st.selectbox("Gospodarz (A)", df['Team'], index=0)
with c2: a_team = st.selectbox("Gość (B)", df['Team'], index=1)

# --- OBLICZENIA (MODEL EXCELOWY) ---
h = df[df['Team'] == h_team].iloc[0]
a = df[df['Team'] == a_team].iloc[0]

# 1. Sumy Goli (Ważone)
h_atk_sum = (h['H_GF']*w_dom + h['T_GF']*w_sezon + h['HxG_F']*w_xg_dom + h['TxG_F']*w_xg_sezon)
h_def_sum = (h['H_GA']*w_dom + h['T_GA']*w_sezon + h['HxG_A']*w_xg_dom + h['TxG_A']*w_xg_sezon)

a_atk_sum = (a['A_GF']*w_dom + a['T_GF']*w_sezon + a['AxG_F']*w_xg_dom + a['TxG_F']*w_xg_sezon)
a_def_sum = (a['A_GA']*w_dom + a['T_GA']*w_sezon + a['AxG_A']*w_xg_dom + a['TxG_A']*w_xg_sezon)

# 2. Współczynniki Siły
h_atk_strength = h_atk_sum / avg_h_gf
h_def_strength = h_def_sum / avg_a_gf
a_atk_strength = a_atk_sum / avg_a_gf
a_def_strength = a_def_sum / avg_h_gf

# 3. Lambda (Oczekiwane Gole)
lamb = h_atk_strength * a_def_strength * avg_h_gf
mu = a_atk_strength * h_def_strength * avg_a_gf

# --- INTERFEJS W STYLU EXCEL ---
col_stats1, col_stats2 = st.columns(2)
with col_stats1:
    st.write("### 📊 Współczynniki siły")
    stats_df = pd.DataFrame({
        'Drużyna': [h_team, a_team],
        'Atak (im więcej tym lepiej)': [round(h_atk_strength, 2), round(a_atk_strength, 2)],
        'Obrona (im mniej tym lepiej)': [round(h_def_strength, 2), round(a_def_strength, 2)]
    })
    st.table(stats_df)

with col_stats2:
    st.write("### 🎯 Oczekiwane gole")
    st.metric(f"Gole {h_team}", round(lamb, 2))
    st.metric(f"Gole {a_team}", round(mu, 2))

# --- MACIERZ POISSONA ---
max_g = 6
matrix = np.outer(poisson.pmf(range(max_g), lamb), poisson.pmf(range(max_g), mu))

st.write("### 🟦 Macierz % szans na wynik (Gospodarz vs Gość)")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(matrix, annot=True, fmt=".2%", cmap="YlGnBu", cbar=False,
            xticklabels=range(max_g), yticklabels=range(max_g))
plt.xlabel(f"Gole: {a_team} (Gość)")
plt.ylabel(f"Gole: {h_team} (Gospodarz)")
st.pyplot(fig)

# --- SZANSE PROCENTOWE ---
win_h = np.sum(np.tril(matrix, -1))
draw = np.sum(np.diag(matrix))
win_a = np.sum(np.triu(matrix, 1))

res1, resx, res2 = st.columns(3)
res1.metric(f"Wygrana {h_team}", f"{win_h:.1%}", f"Kurs: {1/win_h:.2f}")
resx.metric("Remis", f"{draw:.1%}", f"Kurs: {1/draw:.2f}")
res2.metric(f"Wygrana {a_team}", f"{win_a:.1%}", f"Kurs: {1/win_a:.2f}")
