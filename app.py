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

# --- SIDEBAR: WAGI ---
st.sidebar.header("⚖️ Konfiguracja Wag")

D_W = [40, 25, 20, 15]
options = [i for i in range(0, 105, 5)]

if 'w_xg_dv' not in st.session_state: st.session_state.w_xg_dv = D_W[0]
if 'w_g_dv' not in st.session_state: st.session_state.w_g_dv = D_W[1]
if 'w_xg_all' not in st.session_state: st.session_state.w_xg_all = D_W[2]
if 'w_g_all' not in st.session_state: st.session_state.w_g_all = D_W[3]

if st.sidebar.button("🔄 Resetuj wagi (40/25/20/15)"):
    st.session_state.w_xg_dv, st.session_state.w_g_dv = D_W[0], D_W[1]
    st.session_state.w_xg_all, st.session_state.w_g_all = D_W[2], D_W[3]
    st.rerun()

v0 = st.sidebar.selectbox("🎯 xG Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_xg_dv), key='w_xg_dv')
v1 = st.sidebar.selectbox("⚽ Gole Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_g_dv), key='w_g_dv')
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(st.session_state.w_xg_all), key='w_xg_all')
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(st.session_state.w_g_all), key='w_g_all')

w_xg_dv, w_g_dv, w_xg_all, w_g_all = v0/100, v1/100, v2/100, v3/100
total_pct = v0 + v1 + v2 + v3
color = "green" if total_pct == 100 else "red"
st.sidebar.markdown(f"### Suma: :{color}[{total_pct}%]")

if total_pct != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

# --- INTERFEJS GŁÓWNY ---
st.title("⚽ Bundesliga Predictor Pro")
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=120)
with c2:
    a_team = st.selectbox("Gość", df['Team'], index=11)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=120)

# --- OBLICZENIA POISSONA ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

lambda_h_raw = (h['HxG_F']*w_xg_dv + h['H_GF']*w_g_dv + h['TxG_F']*w_xg_all + h['T_GF']*w_g_all)
mu_h_raw = (h['HxG_A']*w_xg_dv + h['H_GA']*w_g_dv + h['TxG_A']*w_xg_all + h['T_GA']*w_g_all)
lambda_a_raw = (a['AxG_F']*w_xg_dv + a['A_GF']*w_g_dv + a['TxG_F']*w_xg_all + a['T_GF']*w_g_all)
mu_a_raw = (a['AxG_A']*w_xg_dv + a['A_GA']*w_g_dv + a['TxG_A']*w_xg_all + a['T_GA']*w_g_all)

h_atk_s, h_def_s = (lambda_h_raw / avg_h_gf), (mu_h_raw / avg_a_gf)
a_atk_s, a_def_s = (lambda_a_raw / avg_a_gf), (mu_a_raw / avg_h_gf)

lambda_final = h_atk_s * a_def_s * avg_h_gf
mu_final = a_atk_s * h_def_s * avg_a_gf

max_goals = 12 # Zwiększony bufor dla obliczeń
matrix = np.outer(poisson.pmf(range(max_goals), lambda_final), poisson.pmf(range(max_goals), mu_final))
p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- PROGNOZA ---
st.divider()
st.subheader("🎯 Prognoza Wyniku Końcowego")
m1, mx, m2 = st.columns(3)
m1.metric(label=f"Wygrana {h_team}", value=f"{p1:.1%}", delta=f"Kurs: {1/p1:.2f}")
mx.metric(label="Remis", value=f"{px:.1%}", delta=f"Kurs: {1/px:.2f}")
m2.metric(label=f"Wygrana {a_team}", value=f"{p2:.1%}", delta=f"Kurs: {1/p2:.2f}")

# --- TABELA WSPÓŁCZYNNIKÓW ---
st.write("### 📊 Współczynniki Siły Drużyn")
def fmt_s(val, is_def=False):
    pct = (val - 1)
    color = "green" if (pct < 0 if is_def else pct > 0) else "red"
    return f":{color}[{val:.2f} ({pct:+.0%})]"

st.markdown(f"""
| Drużyna | Atak (Strength) | Obrona (Strength) | Prognozowane Gole |
| :--- | :--- | :--- | :--- |
| **{h_team}** | {fmt_s(h_atk_s)} | {fmt_s(h_def_s, True)} | **{lambda_final:.2f}** |
| **{a_team}** | {fmt_s(a_atk_s)} | {fmt_s(a_def_s, True)} | **{mu_final:.2f}** |
""")

# --- VALUE BET ---
st.write("### 🏦 Kalkulator Value Bet")
ci1, ci2, ci3 = st.columns(3)
with ci1: bk1 = st.text_input(f"Kurs na {h_team}", placeholder="np. 1.85")
with ci2: bkx = st.text_input("Kurs na X", placeholder="np. 3.40")
with ci3: bk2 = st.text_input(f"Kurs na {a_team}", placeholder="np. 4.50")

def get_v(prob, bk):
    try:
        k = float(bk.replace(',', '.'))
        return f"✅ TAK ({k:.2f})" if k > (1/prob) else f"❌ NIE ({k:.2f})"
    except: return "-"

value_data = {
    "Typ": ["1", "X", "2"],
    "Twoje Szanse": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
    "Kurs Sprawiedliwy": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"],
    "Opłacalność?": [get_v(p1, bk1), get_v(px, bkx), get_v(p2, bk2)]
}
st.table(value_data)

# --- ZIELONA MACIERZY DO 7 GOLI ---
with st.expander("⚽ Analityczna macierz prawdopodobieństwa (Wyniki 0-7)"):
    # Zakres 0-7 to pierwsze 8 indeksów (0,1,2,3,4,5,6,7)
    limit = 8
    m_plot = matrix[:limit, :limit]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    # Paleta YlGn: Yellow -> Green
    sns.heatmap(m_plot, annot=True, fmt=".1%", cmap="YlGn", cbar=False, 
                linewidths=0.5, linecolor='white',
                xticklabels=range(limit), yticklabels=range(limit))
    
    plt.title(f"Prawdopodobieństwo Dokładnego Wyniku (0-7 goli)", pad=20)
    plt.xlabel(f"Gole {a_team}", fontsize=10)
    plt.ylabel(f"Gole {h_team}", fontsize=10)
    
    # Estetyka osi
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    st.pyplot(fig)
