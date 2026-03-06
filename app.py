import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# --- DANE BAZOWE (Bundesliga & Premier League) ---
@st.cache_data
def load_all_data():
    bundesliga = {
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
        'Logo_ID': [27, 16, 24, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    
    premier_league = {
        'Team': ['Arsenal', 'Manchester City', 'Manchester United', 'Aston Villa', 'Chelsea', 'Liverpool', 'Brentford', 'Everton', 'Bournemouth', 'Fulham', 'Sunderland', 'Newcastle', 'Crystal Palace', 'Brighton', 'Leeds', 'Tottenham', 'Nottingham Forest', 'West Ham', 'Burnley', 'Wolves'],
        'H_GF': [2.35, 2.40, 1.92, 1.40, 1.64, 1.85, 1.71, 1.20, 1.40, 1.60, 1.57, 1.86, 1.00, 1.46, 1.46, 1.20, 0.92, 1.21, 1.07, 1.06],
        'H_GA': [0.64, 0.73, 1.14, 1.00, 1.14, 1.14, 1.07, 1.26, 1.00, 1.20, 0.92, 1.60, 1.28, 1.06, 1.33, 1.66, 1.35, 1.92, 1.64, 1.93],
        'T_GF': [1.96, 2.03, 1.75, 1.34, 1.82, 1.65, 1.51, 1.17, 1.51, 1.37, 1.03, 1.44, 1.13, 1.26, 1.27, 1.34, 0.96, 1.20, 1.10, 0.73],
        'T_GA': [0.73, 0.93, 1.37, 1.17, 1.17, 1.34, 1.37, 1.13, 1.58, 1.48, 1.17, 1.48, 1.20, 1.24, 1.65, 1.58, 1.48, 1.86, 2.00, 1.73],
        'HxG_F': [2.05, 2.23, 2.13, 1.36, 2.14, 1.90, 2.07, 1.36, 1.63, 1.39, 1.17, 2.19, 1.94, 1.41, 1.76, 1.24, 1.54, 1.39, 1.03, 1.14],
        'HxG_A': [0.74, 1.07, 1.01, 1.32, 1.54, 1.06, 1.31, 1.44, 0.75, 1.35, 1.46, 1.45, 1.51, 1.31, 1.32, 1.58, 1.59, 1.66, 1.88, 1.73],
        'TxG_F': [1.96, 2.01, 1.91, 1.34, 2.12, 1.86, 1.76, 1.30, 1.71, 1.26, 1.03, 1.63, 1.67, 1.45, 1.51, 1.17, 1.20, 1.29, 0.94, 0.93],
        'TxG_A': [0.79, 1.19, 1.27, 1.54, 1.47, 1.27, 1.47, 1.51, 1.45, 1.58, 1.61, 1.37, 1.50, 1.47, 1.54, 1.55, 1.72, 1.84, 2.16, 1.80],
        'Logo_ID': [11, 281, 985, 405, 631, 31, 1148, 29, 989, 931, 289, 762, 873, 1237, 399, 148, 703, 379, 1132, 543]
    }
    return pd.DataFrame(bundesliga), pd.DataFrame(premier_league)

df_bl, df_epl = load_all_data()

# --- LOGIKA DIXON-COLES ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

def get_prediction(df, h_team, a_team, w, rho):
    avg_h_gf = df['H_GF'].mean()
    avg_a_gf = df['T_GF'].mean() - 0.2 # Szacunkowa różnica dom/wyjazd
    
    h = df[df['Team'] == h_team].iloc[0]
    a = df[df['Team'] == a_team].iloc[0]
    
    l_h_r = (h['HxG_F']*w[0] + h['H_GF']*w[1] + h['TxG_F']*w[2] + h['T_GF']*w[3])
    m_h_r = (h['HxG_A']*w[0] + h['H_GA']*w[1] + h['TxG_A']*w[2] + h['T_GA']*w[3])
    l_a_r = (a['TxG_F']*w[0] + a['T_GF']*w[1] + a['TxG_F']*w[2] + a['T_GF']*w[3]) # Przybliżenie dla wyjazdów
    m_a_r = (a['TxG_A']*w[0] + a['T_GA']*w[1] + a['TxG_A']*w[2] + a['T_GA']*w[3])
    
    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_h_gf)
    a_atk_s, a_def_s = (l_a_r / avg_h_gf), (m_a_r / avg_h_gf)
    
    lambda_f = h_atk_s * a_def_s * avg_h_gf
    mu_f = a_atk_s * h_def_s * avg_a_gf
    
    matrix = np.zeros((12, 12))
    for x in range(12):
        for y in range(12):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, rho)
    matrix /= matrix.sum()
    
    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    return p1, px, p2, lambda_f, mu_f, h_atk_s, h_def_s, a_atk_s, a_def_s, matrix

# --- INTERFEJS ---
tab1, tab2 = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League"])

def run_league_ui(df, league_name, upcoming_matches):
    # Sidebar
    st.sidebar.header(f"⚙️ Konfiguracja {league_name}")
    rho = st.sidebar.slider(f"rho ({league_name})", 0.0, 0.3, 0.1, 0.01)
    
    if f'reset_{league_name}' not in st.session_state: st.session_state[f'reset_{league_name}'] = 0
    if st.sidebar.button(f"Reset wag {league_name}"): st.session_state[f'reset_{league_name}'] += 1
    
    opts = [i for i in range(0, 105, 5)]
    w0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", opts, index=opts.index(45), key=f"w0_{league_name}_{st.session_state[f'reset_{league_name}']}")
    w1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", opts, index=opts.index(30), key=f"w1_{league_name}_{st.session_state[f'reset_{league_name}']}")
    w2 = st.sidebar.selectbox("📊 xG Cały Sezon %", opts, index=opts.index(15), key=f"w2_{league_name}_{st.session_state[f'reset_{league_name}']}")
    w3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", opts, index=opts.index(10), key=f"w3_{league_name}_{st.session_state[f'reset_{league_name}']}")
    
    if (w0+w1+w2+w3) != 100: st.error("Suma wag musi być 100%!"); st.stop()
    weights = [w0/100, w1/100, w2/100, w3/100]

    # Mecz
    st.title(f"⚽ {league_name} Predictor")
    c_h, c_a = st.columns(2)
    with c_h: h_sel = st.selectbox("Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
    with c_a: a_sel = st.selectbox("Gość", df['Team'], index=1, key=f"a_{league_name}")
    
    p1, px, p2, lf, mf, has, hds, aas, ads, mtx = get_prediction(df, h_sel, a_sel, weights, rho)
    
    # Rezultaty 1X2
    st.divider()
    m1, mx, m2 = st.columns(3)
    m1.metric(h_sel, f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
    mx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
    m2.metric(a_sel, f"{p2:.1%}", f"Kurs: {1/p2:.2f}")
    
    # Współczynniki
    st.write("### 📊 Współczynniki Siły")
    def fmt(v, d=False):
        c = "green" if (v<1 if d else v>1) else "red"
        return f":{c}[{v:.2f} ({v-1:+.0%})]"
    st.markdown(f"| Drużyna | Atak | Obrona | Gole Exp. |\n| :--- | :--- | :--- | :--- |\n| **{h_sel}** | {fmt(has)} | {fmt(hds,True)} | **{lf:.2f}** |\n| **{a_sel}** | {fmt(aas)} | {fmt(ads,True)} | **{mf:.2f}** |")

    # Macierz
    st.write("### ⚽ Macierz Prawdopodobieństwa")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mtx[:8, :8], annot=True, fmt=".1%", cmap="YlGn", cbar=False)
    plt.xlabel(f"Gole {a_sel}"); plt.ylabel(f"Gole {h_sel}")
    st.pyplot(fig)

    # Terminarz
    st.divider()
    st.subheader("📅 Terminarz i Kursy Fair")
    res = []
    for ht, at in upcoming_matches:
        tp1, tpx, tp2, _, _, _, _, _, _, _ = get_prediction(df, ht, at, weights, rho)
        res.append({"Mecz": f"{ht} [{1/tp1:.2f}] - {at} [{1/tp2:.2f}]", "Remis [X]": f"{1/tpx:.2f}"})
    st.table(pd.DataFrame(res))

# --- URUCHOMIENIE ---
with tab1:
    bl_matches = [("Borussia Dortmund", "FC Cologne"), ("RB Leipzig", "Bayern Munich"), ("Union Berlin", "Augsburg")]
    run_league_ui(df_bl, "Bundesliga", bl_matches)

with tab2:
    epl_matches = [("Liverpool", "Everton"), ("Arsenal", "Manchester United"), ("Manchester City", "Chelsea"), ("Tottenham", "Aston Villa")]
    run_league_ui(df_epl, "Premier League", epl_matches)
