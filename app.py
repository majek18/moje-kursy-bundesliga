import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Predictor Pro", layout="wide")

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

# --- SIDEBAR: KONFIGURACJA WAG (SELECTBOXY) ---
st.sidebar.header("⚖️ Konfiguracja Wag")

# Opcje co 10 punktów procentowych
options = [i for i in range(0, 110, 10)]
# Mapowanie: 40 -> index 4, 30 -> index 3 itd.

if 'sel0' not in st.session_state: st.session_state.sel0 = 40
if 'sel1' not in st.session_state: st.session_state.sel1 = 30
if 'sel2' not in st.session_state: st.session_state.sel2 = 20
if 'sel3' not in st.session_state: st.session_state.sel3 = 10

if st.sidebar.button("🔄 Resetuj wagi (40/30/20/10)"):
    st.session_state.sel0 = 40
    st.session_state.sel1 = 30
    st.session_state.sel2 = 20
    st.session_state.sel3 = 10
    st.rerun()

v0 = st.sidebar.selectbox("🏠 Gole Dom/Wyjazd (%)", options, index=options.index(st.session_state.sel0), key='sel0')
v1 = st.sidebar.selectbox("🌍 Gole Cały Sezon (%)", options, index=options.index(st.session_state.sel1), key='sel1')
v2 = st.sidebar.selectbox("✈️ xG Dom/Wyjazd (%)", options, index=options.index(st.session_state.sel2), key='sel2')
v3 = st.sidebar.selectbox("📈 xG Cały Sezon (%)", options, index=options.index(st.session_state.sel3), key='sel3')

# Konwersja na ułamki do obliczeń
w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100

total_pct = v0 + v1 + v2 + v3
color = "green" if total_pct == 100 else "red"
st.sidebar.markdown(f"### Suma: :{color}[{total_pct}%]")

if total_pct != 100:
    st.sidebar.error("Suma wag musi wynosić dokładnie 100%!")
    st.stop()

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor Pro")
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=100)
with c2:
    a_team = st.selectbox("Gość", df['Team'], index=11)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=100)

# --- OBLICZENIA SIŁY ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

h_atk_r = (h['H_GF']*w0 + h['T_GF']*w1 + h['HxG_F']*w2 + h['TxG_F']*w3)
h_def_r = (h['H_GA']*w0 + h['T_GA']*w1 + h['HxG_A']*w2 + h['TxG_A']*w3)
a_atk_r = (a['A_GF']*w0 + a['T_GF']*w1 + a['AxG_F']*w2 + a['TxG_F']*w3)
a_def_r = (a['A_GA']*w0 + a['T_GA']*w1 + a['AxG_A']*w2 + a['TxG_A']*w3)

h_atk_s, h_def_s = (h_atk_r / avg_h_gf), (h_def_r / avg_a_gf)
a_atk_s, a_def_s = (a_atk_r / avg_a_gf), (a_def_r / avg_h_gf)

lambda_h = h_atk_s * a_def_s * avg_h_gf
mu_a = a_atk_s * h_def_s * avg_a_gf

# --- TABELA SIŁY ---
st.divider()
st.write("### 📊 Współczynniki Siły (vs Średnia Ligowa)")
def fmt_s(val, is_def=False):
    pct = (val - 1)
    color = "green" if (pct < 0 if is_def else pct > 0) else "red"
    return f":{color}[{val:.2f} ({pct:+.0%})]"

st.markdown(f"""
| Drużyna | Atak (Strength) | Obrona (Strength) | Prognozowane Gole |
| :--- | :--- | :--- | :--- |
| **{h_team}** | {fmt_s(h_atk_s)} | {fmt_s(h_def_s, True)} | **{lambda_h:.2f}** |
| **{a_team}** | {fmt_s(a_atk_s)} | {fmt_s(a_def_s, True)} | **{mu_a:.2f}** |
""")

# --- ANALIZA VALUE ---
matrix = np.outer(poisson.pmf(range(7), lambda_h), poisson.pmf(range(7), mu_a))
matrix /= matrix.sum()
p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

st.write("### 🏦 Analiza Opłacalności (Value Bet)")
cin1, cin2, cin3 = st.columns(3)
with cin1: bk1 = st.text_input(f"Kurs na {h_team} (1)", placeholder="np. 1.45")
with cin2: bkx = st.text_input("Kurs na Remis (X)", placeholder="np. 4.20")
with cin3: bk2 = st.text_input(f"Kurs na {a_team} (2)", placeholder="np. 6.50")

def check_v(prob, b_k):
    try:
        bk = float(b_k.replace(',', '.'))
        my_k = 1/prob
        if bk > my_k: return f"✅ TAK ({bk:.2f})"
        return f"❌ NIE ({bk:.2f})"
    except: return "-"

res_df = pd.DataFrame({
    "Typ": ["1 (Gospodarz)", "X (Remis)", "2 (Gość)"],
    "Szanse Modelu": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
    "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"],
    "Value?": [check_v(p1, bk1), check_v(px, bkx), check_v(p2, bk2)]
})
st.table(res_df)

with st.expander("Zobacz macierz prawdopodobieństwa"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False)
    plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
    st.pyplot(fig)
