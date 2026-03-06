import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Predictor Pro (Dixon-Coles)", layout="wide")

# Custom CSS dla lepszego wyglądu
st.markdown("""
    <style>
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #3d414d; }
    .calc-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_stdio=True)

# --- DANE BAZOWE ---
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

# Stałe z Twojego Excela
BASE_H = 1.75
BASE_A = 1.41

# --- FUNKCJA DIXONA-COLESA ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SIDEBAR: KONFIGURACJA ---
st.sidebar.header("⚙️ Parametry Modelu")
rho = st.sidebar.slider("Dixon-Coles (Dixon)", 0.0, 0.2, 0.1, 0.01)

st.sidebar.subheader("⚖️ Wagi Obliczeń")
if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0

options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon (D/W) %", options, index=options.index(45))
v1 = st.sidebar.selectbox("⚽ Gole Sezon (D/W) %", options, index=options.index(30))
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(15))
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(10))

total_pct = v0 + v1 + v2 + v3
if total_pct != 100:
    st.sidebar.error(f"Suma wag: {total_pct}% (musi być 100%)")
    st.stop()

# --- LOGIKA OBLICZEŃ ---
w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100

st.title("⚽ Bundesliga Predictor Pro")
st.markdown(f"Model Dixon-Coles z synchronizacją Excel (Baza: H={BASE_H} / A={BASE_A})")

col_a, col_b = st.columns(2)
with col_a:
    h_team = st.selectbox("Gospodarz (Drużyna A)", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=80)

with col_b:
    a_team = st.selectbox("Gość (Drużyna B)", df['Team'], index=11)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=80)

h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

# 1. Sumy goli (Kropka w kropkę z Excela)
h_atk_sum = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3)
h_def_sum = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3)
a_atk_sum = (a['TxG_F']*w0 + a['T_GF']*w1 + a['HxG_F']*w2 + h['H_GF']*w3) # Odwrócenie dla wyjazdu
a_def_sum = (a['TxG_A']*w0 + a['T_GA']*w1 + a['HxG_A']*w2 + h['H_GA']*w3)

# 2. Współczynniki siły
h_strength_atk = h_atk_sum / BASE_H
h_strength_def = h_def_sum / BASE_A
a_strength_atk = a_atk_sum / BASE_A
a_strength_def = a_def_sum / BASE_H

# 3. Surowa Lambda (Wynik końcowy dla Poissona)
lambda_h = h_strength_atk * a_strength_def * BASE_H
lambda_a = a_strength_atk * h_strength_def * BASE_A

# --- TRANSPARENTNE OBLICZENIA (TWOJA PROŚBA) ---
with st.expander("🔍 Zobacz transparentne obliczenia (Krok po kroku)"):
    st.markdown("### Krok 1: Wyliczenie 'Sumy Goli' na podstawie wag")
    st.latex(rf"Atak_{{sum}} = ({h['HxG_F']} \cdot {w0}) + ({h['H_GF']} \cdot {w1}) + ({h['TxG_F']} \cdot {w2}) + ({h['T_GF']} \cdot {w3}) = {h_atk_sum:.2f}")
    
    st.markdown("### Krok 2: Wyliczenie 'Współczynnika Siły'")
    st.info(f"Dzielimy sumę goli przez stałą statystyczną (Dla gospodarza: {BASE_H}, Dla gościa: {BASE_A})")
    col1, col2 = st.columns(2)
    col1.write(f"**{h_team} Atak:** {h_atk_sum:.2f} / {BASE_H} = **{h_strength_atk:.2f}**")
    col2.write(f"**{a_team} Obrona:** {a_def_sum:.2f} / {BASE_H} = **{a_strength_def:.2f}**")

    st.markdown("### Krok 3: Wyliczenie 'Surowej Lambdy'")
    st.latex(rf"\lambda_{{Gospodarz}} = Atak_{{H}} \cdot Obrona_{{A}} \cdot {BASE_H} = {h_strength_atk:.2f} \cdot {a_strength_def:.2f} \cdot {BASE_H} = {lambda_h:.2f}")

# --- WYNIKI 1X2 ---
st.divider()
max_g = 10
matrix = np.zeros((max_g, max_g))
for x in range(max_g):
    for y in range(max_g):
        p = poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a)
        matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_h, lambda_a, rho)
matrix /= matrix.sum()

p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

res1, resx, res2 = st.columns(3)
res1.metric(f"1 ({h_team})", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
resx.metric("X (Remis)", f"{px:.1%}", f"Kurs: {1/px:.2f}")
res2.metric(f"2 ({a_team})", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

# --- WIZUALIZACJA ---
st.write("### ⚽ Macierz wyników (Zgodna z Excel)")
limit = 6
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".2%", cmap="Blues", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)

# --- DODATKI ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("📉 Under / Over")
    for line in [1.5, 2.5, 3.5]:
        u_p = sum(matrix[x,y] for x in range(max_g) for y in range(max_g) if x+y < line)
        st.write(f"**{line}:** Under {u_p:.1%} ({1/u_p:.2f}) | Over {1-u_p:.1%} ({1/(1-u_p):.2f})")

with c2:
    st.subheader("🏦 Value Bet Checker")
    user_k = st.number_input("Wpisz kurs bukmachera", value=2.0, step=0.1)
    fair_k = 1/p1 # Przykład dla gospodarza
    if user_k > fair_k:
        st.success(f"Value znalezione! (Fair: {fair_k:.2f})")
    else:
        st.error(f"Brak value. (Fair: {fair_k:.2f})")
