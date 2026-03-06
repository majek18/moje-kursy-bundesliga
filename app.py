import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Pro Analyzer", layout="wide")

# --- DANE BUNDESLIGI + LOGOTYPY ---
@st.cache_data
def load_data():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], # ID do logotypów
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
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38]
    }
    return pd.DataFrame(data)

df = load_data()

# --- ŚREDNIE LIGOWE (Podstawy Poissona) ---
AVG_H_GF = df['H_GF'].mean() # Średnia goli strzelonych przez gospodarzy
AVG_A_GF = df['A_GF'].mean() # Średnia goli strzelonych przez gości

# --- MECHANIZM WAG I PRZYCISK RESETU ---
st.sidebar.header("⚖️ Konfiguracja Wag")

DEFAULT_W = [0.40, 0.30, 0.20, 0.10]

if 'w_dom' not in st.session_state: st.session_state['w_dom'] = DEFAULT_W[0]
if 'w_sezon' not in st.session_state: st.session_state['w_sezon'] = DEFAULT_W[1]
if 'w_xg_dom' not in st.session_state: st.session_state['w_xg_dom'] = DEFAULT_W[2]
if 'w_xg_sezon' not in st.session_state: st.session_state['w_xg_sezon'] = DEFAULT_W[3]

def reset_weights():
    st.session_state['w_dom'] = DEFAULT_W[0]
    st.session_state['w_sezon'] = DEFAULT_W[1]
    st.session_state['w_xg_dom'] = DEFAULT_W[2]
    st.session_state['w_xg_sezon'] = DEFAULT_W[3]

def sync_weights(changed_key):
    keys = ['w_dom', 'w_sezon', 'w_xg_dom', 'w_xg_sezon']
    remaining_keys = [k for k in keys if k != changed_key]
    new_val = st.session_state[changed_key]
    old_total_others = sum(st.session_state[k] for k in remaining_keys)
    target_others = 1.0 - new_val
    if old_total_others > 0:
        for k in remaining_keys:
            st.session_state[k] = (st.session_state[k] / old_total_others) * target_others
    else:
        for k in remaining_keys: st.session_state[k] = target_others / 3

st.sidebar.slider("🏠 Gole Dom/Wyjazd", 0.0, 1.0, key='w_dom', on_change=sync_weights, args=('w_dom',))
st.sidebar.slider("🌍 Gole Cały Sezon", 0.0, 1.0, key='w_sezon', on_change=sync_weights, args=('w_sezon',))
st.sidebar.slider("✈️ xG Dom/Wyjazd", 0.0, 1.0, key='w_xg_dom', on_change=sync_weights, args=('w_xg_dom',))
st.sidebar.slider("📈 xG Cały Sezon", 0.0, 1.0, key='w_xg_sezon', on_change=sync_weights, args=('w_xg_sezon',))

if st.sidebar.button("🔄 Resetuj do domyślnych"):
    reset_weights()
    st.rerun()

# --- WYBÓR MECZU Z HERBAMI ---
st.title("🚀 Bundesliga Predictor Pro")
c1, c2 = st.columns(2)

with c1:
    h_team = st.selectbox("Gospodarz (A)", df['Team'], index=0)
    st.image(f"https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/bundesliga/{h_team.replace(' ', '%20')}.png", width=100)

with c2:
    a_team = st.selectbox("Gość (B)", df['Team'], index=1)
    st.image(f"https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/bundesliga/{a_team.replace(' ', '%20')}.png", width=100)

# --- OBLICZENIA (POISSON ZE ŚREDNIMI) ---
h = df[df['Team'] == h_team].iloc[0]
a = df[df['Team'] == a_team].iloc[0]
w = st.session_state

# 1. Obliczanie siły ataku/obrony gospodarza i gościa
h_atk_raw = (h['H_GF']*w.w_dom + h['T_GF']*w.w_sezon + h['HxG_F']*w.w_xg_dom + h['TxG_F']*w.w_xg_sezon)
h_def_raw = (h['H_GA']*w.w_dom + h['T_GA']*w.w_sezon + h['HxG_A']*w.w_xg_dom + h['TxG_A']*w.w_xg_sezon)

a_atk_raw = (a['A_GF']*w.w_dom + a['T_GF']*w.w_sezon + a['AxG_F']*w.w_xg_dom + a['TxG_F']*w.w_xg_sezon)
a_def_raw = (a['A_GA']*w.w_dom + a['T_GA']*w.w_sezon + a['AxG_A']*w.w_xg_dom + a['TxG_A']*w.w_xg_sezon)

# 2. Współczynniki (SIŁA / ŚREDNIA ODPOWIEDNIEJ STRONY)
h_atk_s = (h_atk_raw / AVG_H_GF) - 1
h_def_s = (h_def_raw / AVG_A_GF) - 1 # Bronisz przeciwko gościom, więc dzielisz przez ich średnią
a_atk_s = (a_atk_raw / AVG_A_GF) - 1
a_def_s = (a_def_raw / AVG_H_GF) - 1 # Bronisz przeciwko gospodarzom

# 3. Ostateczne xG (Lambda i Mu)
lambda_h = (h_atk_s + 1) * (a_def_s + 1) * AVG_H_GF
mu_a = (a_atk_s + 1) * (h_def_s + 1) * AVG_A_GF

# --- WYNIKI I TABELA ---
st.divider()
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.write("### 📊 Siła względem średniej ligi")
    def color_val(val, type='atk'):
        color = "green" if (val > 0 if type == 'atk' else val < 0) else "red"
        return f":{color}[{val:+.1%}]"
    
    st.markdown(f"""
    | Drużyna | Atak | Obrona |
    | :--- | :--- | :--- |
    | **{h_team}** | {color_val(h_atk_s, 'atk')} | {color_val(h_def_s, 'def')} |
    | **{a_team}** | {color_val(a_atk_s, 'atk')} | {color_val(a_def_s, 'def')} |
    """)

with col_res2:
    st.write("### 🎯 Prognozowane gole (xG meczowe)")
    st.metric(f"Oczekiwane gole {h_team}", f"{lambda_h:.2f}")
    st.metric(f"Oczekiwane gole {a_team}", f"{mu_a:.2f}")

# --- MACIERZ POISSONA ---
matrix = np.outer(poisson.pmf(range(7), lambda_h), poisson.pmf(range(7), mu_a))
matrix /= matrix.sum()

st.write("### 🟦 Macierz szans na dokładny wynik")
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)

wh, dr, wa = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
c_h, c_x, c_a = st.columns(3)
c_h.metric(f"Wygrana {h_team}", f"{wh:.1%}", f"Kurs: {1/wh:.2f}")
c_x.metric("Remis", f"{dr:.1%}", f"Kurs: {1/dr:.2f}")
c_a.metric(f"Wygrana {a_team}", f"{wa:.1%}", f"Kurs: {1/wa:.2f}")
