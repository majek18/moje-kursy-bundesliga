import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURACJA STRONY ---
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
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.06, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38],
        'Logo_ID': [27, 16, 24, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

df = load_data()
avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()

# --- FUNKCJA DIXONA-COLESA ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfiguracja")
rho = st.sidebar.slider("Parametr Dixon-Coles (rho)", 0.0, 0.3, 0.1, 0.01)

if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0
def reset_weights(): st.session_state.reset_counter += 1

st.sidebar.button("🔄 Resetuj wagi (40/25/20/15)", on_click=reset_weights)
options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", options, index=options.index(40), key=f"w0_{st.session_state.reset_counter}")
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", options, index=options.index(25), key=f"w1_{st.session_state.reset_counter}")
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", options, index=options.index(20), key=f"w2_{st.session_state.reset_counter}")
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, index=options.index(15), key=f"w3_{st.session_state.reset_counter}")

total_pct = v0 + v1 + v2 + v3
color = "green" if total_pct == 100 else "red"
st.sidebar.markdown(f"### Suma: :{color}[{total_pct}%]")
if total_pct != 100: st.sidebar.error("Suma wag musi wynosić 100%!"); st.stop()

# --- LOGIKA ---
w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100
st.title("⚽ Bundesliga Predictor Pro (Dixon-Coles)")

col_a, col_b = st.columns(2)
with col_a:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=100)
with col_b:
    a_team = st.selectbox("Gość", df['Team'], index=1)
    a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=100)

h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3)
m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3)
l_a_r = (a['AxG_F']*w0 + a['A_GF']*w1 + a['TxG_F']*w2 + a['T_GF']*w3)
m_a_r = (a['AxG_A']*w0 + a['A_GA']*w1 + a['TxG_A']*w2 + a['T_GA']*w3)

h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)

lambda_f = h_atk_s * a_def_s * avg_h_gf
mu_f = a_atk_s * h_def_s * avg_a_gf

# --- STEP BY STEP ---
with st.expander("🧮 Ścieżka Obliczeniowa (Jak powstał wynik?)"):
    st.subheader("Krok 1: Średnie ważone i Siła")
    st.latex(rf"\text{{Siła Ataku }} {h_team} = {h_atk_s:.2f} \quad | \quad \text{{Siła Obrony }} {a_team} = {a_def_s:.2f}")
    st.latex(rf"\lambda (Gospodarz) = {h_atk_s:.2f} \cdot {a_def_s:.2f} \cdot {avg_h_gf:.2f} = \mathbf{{{lambda_f:.2f}}}")
    st.latex(rf"\mu (Gość) = {a_atk_s:.2f} \cdot {h_def_s:.2f} \cdot {avg_a_gf:.2f} = \mathbf{{{mu_f:.2f}}}")

    st.subheader("Krok 2: Rozkład Poissona (Możliwe gole)")
    st.write("Wykorzystujemy wzór na prawdopodobieństwo uzyskania dokładnie $k$ goli:")
    st.latex(r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}")
    
    goals_range = range(6)
    h_probs = [poisson.pmf(k, lambda_f) for k in goals_range]
    a_probs = [poisson.pmf(k, mu_f) for k in goals_range]
    
    prob_df = pd.DataFrame({
        "Gole": goals_range,
        f"{h_team} (%)": [f"{p:.2%}" for p in h_probs],
        f"{a_team} (%)": [f"{p:.2%}" for p in a_probs]
    }).set_index("Gole")
    st.table(prob_df.T)

# --- MACIERZ I OBLICZENIA KOŃCOWE ---
max_g = 12
matrix = np.zeros((max_g, max_g))
for x in range(max_g):
    for y in range(max_g):
        p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
        matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, rho)
matrix /= matrix.sum()

p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- WIDOK 1X2 ---
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
c2.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
c3.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

# --- MACIERZ GRAFICZNA ---
st.write("### ⚽ Macierz Prawdopodobieństwa (0-7 goli)")
limit = 8
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".1%", cmap="YlGn", cbar=False)
plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
st.pyplot(fig)

# --- ANALIZA U/O ---
st.divider()
st.subheader("📉 Analiza Under / Over")
lines = [1.5, 2.5, 3.5, 4.5]
ou_cols = st.columns(len(lines))
for i, line in enumerate(lines):
    u_p = sum(matrix[x, y] for x in range(max_g) for y in range(max_g) if x + y < line)
    o_p = 1 - u_p
    with ou_cols[i]:
        st.markdown(f"**Linia {line}**")
        st.write(f"🟢 **OVER**: {o_p:.1%} (k: {1/o_p:.2f})")
        st.write(f"🔴 **UNDER**: {u_p:.1%} (k: {1/u_p:.2f})")
        st.progress(o_p)

# --- VALUE BET ---
st.divider()
st.write("### 🏦 Kalkulator Value Bet")
v1, v2, v3 = st.columns(3)
with v1: bk1 = st.text_input(f"Kurs {h_team}", "2.00")
with v2: bkx = st.text_input("Kurs X", "3.40")
with v3: bk2 = st.text_input(f"Kurs {a_team}", "4.00")

def check_v(prob, bk):
    try:
        k = float(bk.replace(',', '.'))
        return f"✅ TAK ({k:.2f})" if k > (1/prob) else f"❌ NIE ({k:.2f})"
    except: return "-"

st.table({
    "Typ": ["1", "X", "2"],
    "Model (%)": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
    "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"],
    "Value?": [check_v(p1, bk1), check_v(px, bkx), check_v(p2, bk2)]
})
