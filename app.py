import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Bundesliga Predictor Pro", layout="wide")

# POPRAWIONY BLOK CSS (Naprawia błąd TypeError z linii 11/12)
st.markdown("""
<style>
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d414d;
    }
    .calc-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_stdio=True)

# --- DANE BAZOWE (Zsynchronizowane z Twoim Excelem) ---
@st.cache_data
def load_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        'H_GF': [4.00, 2.33, 2.25, 1.75, 2.25, 2.08, 1.83, 1.91, 1.31, 1.42, 1.46, 1.08, 1.75, 1.08, 1.18, 1.17, 1.58, 1.08],
        'H_GA': [1.00, 0.92, 1.17, 1.00, 1.42, 0.92, 1.50, 1.09, 1.46, 1.42, 1.23, 1.50, 1.58, 1.17, 1.64, 1.75, 2.17, 2.25],
        'T_GF': [3.67, 2.13, 2.04, 2.00, 1.92, 1.88, 2.00, 1.42, 1.25, 1.21, 1.08, 1.13, 1.38, 1.13, 0.96, 1.04, 1.38, 0.92],
        'T_GA': [0.96, 1.04, 1.29, 1.33, 1.38, 1.21, 2.04, 1.63, 1.71, 1.58, 1.46, 1.63, 1.71, 1.63, 1.67, 1.83, 2.21, 2.21],
        'HxG_F': [3.43, 2.00, 2.07, 2.11, 2.65, 2.26, 1.69, 1.86, 1.31, 1.51, 1.59, 1.40, 1.51, 1.92, 1.00, 1.60, 1.52, 1.47],
        'HxG_A': [1.04, 1.23, 1.28, 1.35, 1.51, 0.92, 1.26, 1.07, 1.67, 1.31, 1.58, 1.52, 1.65, 1.53, 1.54, 1.36, 1.84, 2.06],
        'TxG_F': [3.07, 1.85, 1.85, 1.96, 2.20, 2.02, 1.56, 1.42, 1.25, 1.42, 1.32, 1.43, 1.45, 1.63, 0.97, 1.32, 1.41, 1.36],
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
        'Logo_ID': [27, 16, 24, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

df = load_data()

# STAŁE Z EXCELA
BASE_H = 1.75
BASE_A = 1.41

# --- FUNKCJA DIXONA-COLESA ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SIDEBAR: KONFIGURACJA WAG ---
st.sidebar.header("⚙️ Konfiguracja Wag")
rho = st.sidebar.slider("Dixon-Coles (rho)", 0.0, 0.2, 0.1, 0.01)

# Ustawienie domyślnych wag zgodnie z Twoim Resetem (45/30/15/10)
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", range(0, 105, 5), index=9) # 45
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", range(0, 105, 5), index=6) # 30
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", range(0, 105, 5), index=3) # 15
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", range(0, 105, 5), index=2) # 10

if (v0 + v1 + v2 + v3) != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor Pro (Dixon-Coles)")
col_a, col_b = st.columns(2)
with col_a:
    h_team = st.selectbox("Gospodarz (Drużyna A)", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=80)
with col_b:
    a_team = st.selectbox("Gość (Drużyna B)", df['Team'], index=11) # M.Gladbach
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=80)

h = df[df['Team'] == h_team].iloc[0]
a = df[df['Team'] == a_team].iloc[0]

# --- LOGIKA OBLICZEŃ (IDENTYCZNA Z EXCELEM) ---
# 1. Sumy goli
h_atk_sum = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3)
h_def_sum = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3)
a_atk_sum = (a['TxG_F']*w0 + a['T_GF']*w1 + a['HxG_F']*w2 + h['H_GF']*w3)
a_def_sum = (a['TxG_A']*w0 + a['T_GA']*w1 + a['HxG_A']*w2 + h['H_GA']*w3)

# 2. Współczynniki siły (Dzielenie przez stałe bazowe)
h_str_atk = h_atk_sum / BASE_H
h_str_def = h_def_sum / BASE_A
a_str_atk = a_atk_sum / BASE_A
a_str_def = a_def_sum / BASE_H

# 3. Surowa Lambda
lambda_h = h_str_atk * a_str_def * BASE_H
lambda_a = a_str_atk * h_str_def * BASE_A

# --- TRANSPARENTNE OBLICZENIA ---
with st.expander("🔍 Obliczenia krok po kroku (Zgodność z Excel)"):
    st.markdown("### Krok 1: Sumy Goli")
    st.write(f"**{h_team}** -> Atak: {h_atk_sum:.2f} | Obrona: {h_def_sum:.2f}")
    st.write(f"**{a_team}** -> Atak: {a_atk_sum:.2f} | Obrona: {a_def_sum:.2f}")
    
    st.markdown("### Krok 2: Współczynniki Siły")
    st.latex(rf"Atak_{{str\_H}} = {h_atk_sum:.2f} / {BASE_H} = {h_str_atk:.2f}")
    st.latex(rf"Obrona_{{str\_A}} = {a_def_sum:.2f} / {BASE_H} = {a_str_def:.2f}")
    
    st.markdown("### Krok 3: Surowa Lambda")
    st.latex(rf"\lambda_H = {h_str_atk:.2f} \cdot {a_str_def:.2f} \cdot {BASE_H} = {lambda_h:.2f}")

# --- WIDOK WYNIKÓW ---
max_g = 10
matrix = np.zeros((max_g, max_g))
for x in range(max_g):
    for y in range(max_g):
        p = poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a)
        matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_h, lambda_a, rho)
matrix /= matrix.sum()

p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
c2.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
c3.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

# Macierz wizualna
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(matrix[:6, :6], annot=True, fmt=".2%", cmap="YlGnBu", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)
