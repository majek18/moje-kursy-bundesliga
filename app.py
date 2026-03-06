import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Pro Analyzer", layout="wide")

# --- DANE BUNDESLIGI ---
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

# --- WAGI ---
st.sidebar.header("⚖️ Konfiguracja Wag")
DEFAULT_W = [0.40, 0.30, 0.20, 0.10]

# Obsługa przycisku resetu bez wywalania błędu
if st.sidebar.button("🔄 Resetuj do domyślnych"):
    for key in ['w_dom', 'w_sezon', 'w_xg_dom', 'w_xg_sezon']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if 'w_dom' not in st.session_state: st.session_state['w_dom'] = DEFAULT_W[0]
if 'w_sezon' not in st.session_state: st.session_state['w_sezon'] = DEFAULT_W[1]
if 'w_xg_dom' not in st.session_state: st.session_state['w_xg_dom'] = DEFAULT_W[2]
if 'w_xg_sezon' not in st.session_state: st.session_state['w_xg_sezon'] = DEFAULT_W[3]

def sync_weights(changed_key):
    keys = ['w_dom', 'w_sezon', 'w_xg_dom', 'w_xg_sezon']
    remaining = [k for k in keys if k != changed_key]
    new_val = st.session_state[changed_key]
    old_total_others = sum(st.session_state[k] for k in remaining)
    target_others = 1.0 - new_val
    if old_total_others > 0:
        for k in remaining:
            st.session_state[k] = (st.session_state[k] / old_total_others) * target_others
    else:
        for k in remaining: st.session_state[k] = target_others / 3

st.sidebar.slider("🏠 Gole Dom/Wyjazd", 0.0, 1.0, key='w_dom', on_change=sync_weights, args=('w_dom',))
st.sidebar.slider("🌍 Gole Cały Sezon", 0.0, 1.0, key='w_sezon', on_change=sync_weights, args=('w_sezon',))
st.sidebar.slider("✈️ xG Dom/Wyjazd", 0.0, 1.0, key='w_xg_dom', on_change=sync_weights, args=('w_xg_dom',))
st.sidebar.slider("📈 xG Cały Sezon", 0.0, 1.0, key='w_xg_sezon', on_change=sync_weights, args=('w_xg_sezon',))

# --- UI ---
st.title("⚽ Bundesliga Match Predictor")
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
w = st.session_state

h_atk_r = (h['H_GF']*w.w_dom + h['T_GF']*w.w_sezon + h['HxG_F']*w.w_xg_dom + h['TxG_F']*w.w_xg_sezon)
h_def_r = (h['H_GA']*w.w_dom + h['T_GA']*w.w_sezon + h['HxG_A']*w.w_xg_dom + h['TxG_A']*w.w_xg_sezon)
a_atk_r = (a['A_GF']*w.w_dom + a['T_GF']*w.w_sezon + a['AxG_F']*w.w_xg_dom + a['TxG_F']*w.w_xg_sezon)
a_def_r = (a['A_GA']*w.w_dom + a['T_GA']*w.w_sezon + a['AxG_A']*w.w_xg_dom + a['TxG_A']*w.w_xg_sezon)

h_atk_s, h_def_s = (h_atk_r / AVG_H_GF) - 1, (h_def_r / AVG_A_GF) - 1
a_atk_s, a_def_s = (a_atk_r / AVG_A_GF) - 1, (a_def_r / AVG_H_GF) - 1

lambda_h = (h_atk_s + 1) * (a_def_s + 1) * AVG_H_GF
mu_a = (a_atk_s + 1) * (h_def_s + 1) * AVG_A_GF

# --- WIDOK ---
st.divider()
res_c1, res_c2 = st.columns(2)
with res_c1:
    st.write("### 📊 Siła Ataku / Obrony")
    def color_val(v, t='atk'):
        c = "green" if (v > 0 if t=='atk' else v < 0) else "red"
        return f":{c}[{v:+.1%}]"
    st.markdown(f"**{h_team}**: Atak {color_val(h_atk_s)} | Obrona {color_val(h_def_s, 'def')}")
    st.markdown(f"**{a_team}**: Atak {color_val(a_atk_s)} | Obrona {color_val(a_def_s, 'def')}")

with res_c2:
    st.write("### 🎯 Prognozowane xG")
    st.metric(h_team, f"{lambda_h:.2f}")
    st.metric(a_team, f"{mu_a:.2f}")

# Macierz i kursy
matrix = np.outer(poisson.pmf(range(7), lambda_h), poisson.pmf(range(7), mu_a))
matrix /= matrix.sum()

st.write("### 🟦 Macierz prawdopodobieństwa")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="Blues", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)

wh, dr, wa = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
k1, kx, k2 = st.columns(3)
k1.metric("1", f"{wh:.1%}", f"Kurs: {1/wh:.2f}")
kx.metric("X", f"{dr:.1%}", f"Kurs: {1/dr:.2f}")
k2.metric("2", f"{wa:.1%}", f"Kurs: {1/wa:.2f}")
