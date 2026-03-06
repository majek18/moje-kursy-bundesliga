import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Pro Analyzer", layout="wide")

# --- TWOJA BAZA DANYCH (Zgodnie ze screenami) ---
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
AVG_H_GF = df['H_GF'].mean()
AVG_A_GF = df['A_GF'].mean()

# --- SIDEBAR: KONFIGURACJA WAG ---
st.sidebar.header("⚖️ Konfiguracja Wag")
DEFAULT_W = [0.40, 0.30, 0.20, 0.10]

if st.sidebar.button("🔄 Resetuj tylko wagi"):
    for key in ['w0', 'w1', 'w2', 'w3']:
        if key in st.session_state: del st.session_state[key]
    st.rerun()

if 'w0' not in st.session_state: st.session_state.w0 = DEFAULT_W[0]
if 'w1' not in st.session_state: st.session_state.w1 = DEFAULT_W[1]
if 'w2' not in st.session_state: st.session_state.w2 = DEFAULT_W[2]
if 'w3' not in st.session_state: st.session_state.w3 = DEFAULT_W[3]

def sync_weights(changed_key):
    keys = ['w0', 'w1', 'w2', 'w3']
    rem = [k for k in keys if k != changed_key]
    diff = 1.0 - st.session_state[changed_key]
    old_sum = sum(st.session_state[k] for k in rem)
    if old_sum > 0:
        for k in rem: st.session_state[k] = (st.session_state[k] / old_sum) * diff
    else:
        for k in rem: st.session_state[k] = diff / 3

st.sidebar.slider("🏠 Gole Dom/Wyjazd", 0.0, 1.0, key='w0', on_change=sync_weights, args=('w0',))
st.sidebar.slider("🌍 Gole Cały Sezon", 0.0, 1.0, key='w1', on_change=sync_weights, args=('w1',))
st.sidebar.slider("✈️ xG Dom/Wyjazd", 0.0, 1.0, key='w2', on_change=sync_weights, args=('w2',))
st.sidebar.slider("📈 xG Cały Sezon", 0.0, 1.0, key='w3', on_change=sync_weights, args=('w3',))

# --- WYBÓR ZESPOŁÓW I LOGO ---
st.title("⚽ Bundesliga Predictor Pro")
c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=120)

with c2:
    a_team = st.selectbox("Gość", df['Team'], index=1)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=120)

# --- OBLICZENIA SIŁY (TWOJA LOGIKA) ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
ws = [st.session_state.w0, st.session_state.w1, st.session_state.w2, st.session_state.w3]

h_atk_r = (h['H_GF']*ws[0] + h['T_GF']*ws[1] + h['HxG_F']*ws[2] + h['TxG_F']*ws[3])
h_def_r = (h['H_GA']*ws[0] + h['T_GA']*ws[1] + h['HxG_A']*ws[2] + h['TxG_A']*ws[3])
a_atk_r = (a['A_GF']*ws[0] + a['T_GF']*ws[1] + a['AxG_F']*ws[2] + a['TxG_F']*ws[3])
a_def_r = (a['A_GA']*ws[0] + a['T_GA']*ws[1] + a['AxG_A']*ws[2] + a['TxG_A']*ws[3])

# Współczynniki siły (Strength)
h_atk_s, h_def_s = (h_atk_r / AVG_H_GF), (h_def_r / AVG_A_GF)
a_atk_s, a_def_s = (a_atk_r / AVG_A_GF), (a_def_r / AVG_H_GF)

# Prognoza xG na mecz (Lambda i Mu)
lambda_h = h_atk_s * a_def_s * AVG_H_GF
mu_a = a_atk_s * h_def_s * AVG_A_GF

# --- TABELA WSPÓŁCZYNNIKÓW ---
st.divider()
st.write("### 📊 Współczynniki Siły (vs Średnia Ligowa)")
def color_strength(val):
    # Dla ataku: >1 jest zielone, <1 czerwone. Dla obrony: <1 zielone, >1 czerwone.
    # Uproszczone wyświetlanie procentowe
    pct = (val - 1)
    color = "green" if pct > 0 else "red"
    return f":{color}[{val:.2f} ({pct:+.0%})]"

def color_def(val):
    pct = (val - 1)
    color = "green" if pct < 0 else "red"
    return f":{color}[{val:.2f} ({pct:+.0%})]"

st.markdown(f"""
| Drużyna | Atak (Strength) | Obrona (Strength) | Prognozowane Gole |
| :--- | :--- | :--- | :--- |
| **{h_team}** | {color_strength(h_atk_s)} | {color_def(h_def_s)} | **{lambda_h:.2f}** |
| **{a_team}** | {color_strength(a_atk_s)} | {color_def(a_def_s)} | **{mu_a:.2f}** |
""")

# --- MACIERZ I SZANSE ---
matrix = np.outer(poisson.pmf(range(7), lambda_h), poisson.pmf(range(7), mu_a))
matrix /= matrix.sum()

p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- PORÓWNYWARKA KURSÓW ---
st.write("### 🏦 Szanse i Porównanie z Bukmacherami")
k1, kx, k2 = 1/p1, 1/px, 1/p2

# Dane do tabeli
odds_data = {
    "Źródło": ["Twój Model (%)", "Twój Kurs", "STS", "Fortuna", "Superbet"],
    "1": [f"{p1:.1%}", f"{k1:.2f}", "1.85", "1.82", "1.90"],
    "X": [f"{px:.1%}", f"{kx:.2f}", "3.80", "3.90", "3.75"],
    "2": [f"{p2:.1%}", f"{k2:.2f}", "4.20", "4.30", "4.15"]
}
st.table(pd.DataFrame(odds_data))

# --- WIZUALIZACJA MACIERZY ---
st.write("### 🟦 Prawdopodobieństwo Wyniku")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)
