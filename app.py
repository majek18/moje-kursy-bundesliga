import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Predictor Pro (Dixon-Coles)", layout="wide")

# --- DANE BAZOWE ---
@st.cache_data
def load_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        'H_GF': [4.00, 2.33, 2.25, 1.75, 2.25, 2.08, 1.83, 1.91, 1.31, 1.42, 1.46, 1.17, 1.75, 1.08, 1.18, 1.17, 1.58, 1.08],
        'H_GA': [1.00, 0.92, 1.17, 1.00, 1.42, 0.92, 1.50, 1.09, 1.46, 1.42, 1.23, 1.75, 1.58, 1.17, 1.64, 1.75, 2.17, 2.25],
        'T_GF': [3.67, 2.13, 2.04, 2.00, 1.92, 1.88, 2.00, 1.42, 1.25, 1.21, 1.08, 1.13, 1.38, 1.13, 0.96, 1.04, 1.38, 0.92],
        'T_GA': [0.96, 1.04, 1.29, 1.33, 1.38, 1.21, 2.04, 1.63, 1.71, 1.58, 1.46, 1.63, 1.71, 1.63, 1.67, 1.83, 2.21, 2.21],
        'HxG_F': [3.43, 2.00, 2.07, 2.11, 2.65, 2.26, 1.69, 1.86, 1.31, 1.51, 1.59, 1.46, 1.51, 1.92, 1.00, 1.60, 1.52, 1.47],
        'HxG_A': [1.04, 1.23, 1.28, 1.35, 1.51, 0.92, 1.26, 1.07, 1.67, 1.31, 1.58, 1.73, 1.65, 1.53, 1.54, 1.36, 1.84, 2.06],
        'TxG_F': [3.07, 1.85, 1.85, 1.96, 2.20, 2.02, 1.56, 1.42, 1.25, 1.42, 1.32, 1.43, 1.45, 1.63, 0.97, 1.32, 1.41, 1.36],
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.33, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38],
        'Logo_ID': [27, 16, 24, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

df = load_data()
avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

# --- FUNKCJA DIXONA-COLESA ---
def dixon_coles_adjustment(x, y, lambda_h, mu_a, rho):
    """Korekta prawdopodobieństwa dla wyników 0:0, 1:0, 0:1 i 1:1"""
    if x == 0 and y == 0:
        return 1 - (lambda_h * mu_a * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1 + (mu_a * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parametry Modelu")
rho = st.sidebar.slider("Parametr Korekty Dixona-Colesa (rho)", 0.0, 0.3, 0.1, 0.01)

st.sidebar.subheader("⚖️ Wagi Statystyk")
D_W = [40, 25, 20, 15]
options = [i for i in range(0, 105, 5)]

if 'w0' not in st.session_state: st.session_state.w0, st.session_state.w1, st.session_state.w2, st.session_state.w3 = D_W

v0 = st.sidebar.selectbox("🎯 xG Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w0))
v1 = st.sidebar.selectbox("⚽ Gole Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w1))
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(st.session_state.w2))
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(st.session_state.w3))

w_xg_dv, w_g_dv, w_xg_all, w_g_all = v0/100, v1/100, v2/100, v3/100
if (v0 + v1 + v2 + v3) != 100:
    st.sidebar.error("Suma wag musi być 100%!")
    st.stop()

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor (Dixon-Coles)")
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=120)
with c2:
    a_team = st.selectbox("Gość", df['Team'], index=1)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=120)

# --- OBLICZENIA ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

l_h_raw = (h['HxG_F']*w_xg_dv + h['H_GF']*w_g_dv + h['TxG_F']*w_xg_all + h['T_GF']*w_g_all)
m_h_raw = (h['HxG_A']*w_xg_dv + h['H_GA']*w_g_dv + h['TxG_A']*w_xg_all + h['T_GA']*w_g_all)
l_a_raw = (a['AxG_F']*w_xg_dv + a['A_GF']*w_g_dv + a['TxG_F']*w_xg_all + a['T_GF']*w_g_all)
m_a_raw = (a['AxG_A']*w_xg_dv + a['A_GA']*w_g_dv + a['TxG_A']*w_xg_all + a['T_GA']*w_g_all)

h_atk_s, h_def_s = (l_h_raw / avg_h_gf), (m_h_raw / avg_a_gf)
a_atk_s, a_def_s = (l_a_raw / avg_a_gf), (m_a_raw / avg_h_gf)

lambda_final = h_atk_s * a_def_s * avg_h_gf
mu_final = a_atk_s * h_def_s * avg_a_gf

# Generowanie macierzy z korektą
max_g = 12
matrix = np.zeros((max_g, max_g))

for x in range(max_g):
    for y in range(max_g):
        prob = poisson.pmf(x, lambda_final) * poisson.pmf(y, mu_final)
        adj = dixon_coles_adjustment(x, y, lambda_final, mu_final, rho)
        matrix[x, y] = prob * adj

# Normalizacja macierzy (suma musi być 1)
matrix /= matrix.sum()

p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- WYNIKI ---
st.divider()
st.subheader("🎯 Prognoza (Korekta Dixon-Coles)")
m1, mx, m2 = st.columns(3)
m1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
mx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
m2.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

with st.expander("⚽ Macierz Prawdopodobieństwa (0-7 goli)"):
    limit = 8
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".1%", cmap="YlGn", cbar=False, linewidths=0.5)
    plt.title(f"Rozkład goli z uwzględnieniem rho={rho}")
    plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
    st.pyplot(fig)

st.info("💡 Model Dixona-Colesa koryguje niedoszacowanie niskich remisów, co zazwyczaj podnosi kurs na faworyta i urealnia szanse na 0:0 i 1:1.")
