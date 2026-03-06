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
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38]
    }
    return pd.DataFrame(data)

df = load_data()
avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

# --- INTELIGENTNE SUWAKI (SIDEBAR) ---
st.sidebar.header("⚖️ Balans Wag (Suma 100%)")

if 'weights' not in st.session_state:
    st.session_state.weights = [0.40, 0.30, 0.20, 0.10]

def update_weights(index):
    new_val = st.session_state[f'w{index}']
    old_val = st.session_state.weights[index]
    diff = new_val - old_val
    
    other_indices = [i for i in range(4) if i != index]
    other_sum = sum(st.session_state.weights[i] for i in other_indices)
    
    if other_sum > 0:
        for i in other_indices:
            st.session_state.weights[i] -= diff * (st.session_state.weights[i] / other_sum)
    else:
        for i in other_indices:
            st.session_state.weights[i] -= diff / 3
    
    st.session_state.weights[index] = new_val

w_labels = [
    "🏠 Gole Dom/Wyjazd", 
    "🌍 Gole Cały Sezon", 
    "✈️ xG Dom/Wyjazd", 
    "📈 xG Cały Sezon"
]

for i in range(4):
    st.sidebar.slider(w_labels[i], 0.0, 1.0, st.session_state.weights[i], 
                      key=f'w{i}', on_change=update_weights, args=(i,))

weights = st.session_state.weights

# --- WYBÓR MECZU ---
st.title("🚀 Bundesliga Predictor Pro")
c1, c2 = st.columns(2)
with c1: h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with c2: a_team = st.selectbox("Gość", df['Team'], index=1)

# --- OBLICZENIA SIŁY ---
h = df[df['Team'] == h_team].iloc[0]
a = df[df['Team'] == a_team].iloc[0]

h_atk_sum = (h['H_GF']*weights[0] + h['T_GF']*weights[1] + h['HxG_F']*weights[2] + h['TxG_F']*weights[3])
h_def_sum = (h['H_GA']*weights[0] + h['T_GA']*weights[1] + h['HxG_A']*weights[2] + h['TxG_A']*weights[3])
a_atk_sum = (a['A_GF']*weights[0] + a['T_GF']*weights[1] + a['AxG_F']*weights[2] + a['TxG_F']*weights[3])
a_def_sum = (a['A_GA']*weights[0] + a['T_GA']*weights[1] + a['AxG_A']*weights[2] + a['TxG_A']*weights[3])

h_atk_s = (h_atk_sum / avg_h_gf) - 1
h_def_s = (h_def_sum / avg_a_gf) - 1
a_atk_s = (a_atk_sum / avg_a_gf) - 1
a_def_s = (a_def_sum / avg_h_gf) - 1

# --- MINI-TABELA SIŁY ---
st.write("### 📊 Siła zespołów względem średniej ligi")

def color_value(val, type='atk'):
    color = "green" if (val > 0 if type == 'atk' else val < 0) else "red"
    return f":{color}[{val:+.0%}]"

st.markdown(f"""
| Drużyna | Atak (Skuteczność) | Obrona (Szczelność) |
| :--- | :--- | :--- |
| **{h_team}** | {color_value(h_atk_s, 'atk')} | {color_value(h_def_s, 'def')} |
| **{a_team}** | {color_value(a_atk_s, 'atk')} | {color_value(a_def_s, 'def')} |
""")

# --- MODEL POISSONA ---
lamb = (h_atk_s + 1) * (a_def_s + 1) * avg_h_gf
mu = (a_atk_s + 1) * (h_def_s + 1) * avg_a_gf

matrix = np.outer(poisson.pmf(range(6), lamb), poisson.pmf(range(6), mu))
matrix /= matrix.sum()

# --- MACIERZ ---
st.write("### 🟦 Macierz Szans na Wynik")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)

# --- KURSY ---
wh, dr, wa = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
res1, resx, res2 = st.columns(3)
res1.metric(f"Wygrana {h_team}", f"{wh:.1%}", f"Kurs: {1/wh:.2f}")
resx.metric("Remis", f"{dr:.1%}", f"Kurs: {1/dr:.2f}")
res2.metric(f"Wygrana {a_team}", f"{wa:.1%}", f"Kurs: {1/wa:.2f}")
