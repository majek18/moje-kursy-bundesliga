import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# --- BAZA DANYCH (Zrzuty ekranu + Bundesliga) ---
@st.cache_data
def load_data():
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
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22]
    }
    
    premier_league = {
        'Team': ['Arsenal', 'Manchester City', 'Manchester United', 'Aston Villa', 'Chelsea', 'Liverpool', 'Brentford', 'Everton', 'Bournemouth', 'Fulham', 'Sunderland', 'Newcastle United', 'Crystal Palace', 'Brighton', 'Leeds', 'Tottenham', 'Nottingham Forest', 'West Ham', 'Burnley', 'Wolverhampton Wanderers'],
        'H_GF': [2.35, 2.40, 1.92, 1.40, 1.64, 1.85, 1.71, 1.20, 1.40, 1.60, 1.57, 1.86, 1.00, 1.46, 1.46, 1.20, 0.92, 1.21, 1.07, 1.06],
        'H_GA': [0.64, 0.73, 1.14, 1.00, 1.14, 1.14, 1.07, 1.26, 1.00, 1.20, 0.92, 1.60, 1.28, 1.06, 1.33, 1.66, 1.35, 1.92, 1.64, 1.93],
        'T_GF': [1.62, 1.64, 1.60, 1.28, 2.00, 1.46, 1.33, 1.14, 1.64, 1.14, 0.53, 1.00, 1.26, 1.14, 1.07, 1.50, 1.00, 1.20, 1.13, 0.35],
        'T_GA': [0.81, 1.14, 1.60, 1.35, 1.20, 1.53, 1.66, 1.00, 2.21, 1.78, 1.40, 1.35, 1.13, 1.42, 2.00, 1.50, 1.60, 1.80, 2.33, 1.50],
        'HxG_F': [2.05, 2.23, 2.13, 1.36, 2.14, 1.90, 2.07, 1.36, 1.63, 1.39, 1.17, 2.19, 1.94, 1.41, 1.76, 1.24, 1.54, 1.39, 1.03, 1.14],
        'HxG_A': [0.74, 1.07, 1.01, 1.32, 1.54, 1.06, 1.31, 1.44, 0.75, 1.35, 1.46, 1.45, 1.51, 1.31, 1.32, 1.58, 1.59, 1.66, 1.88, 1.73],
        'TxG_F': [1.87, 1.78, 1.70, 1.32, 2.10, 1.81, 1.48, 1.22, 1.79, 1.11, 0.91, 1.03, 1.43, 1.48, 1.23, 1.10, 0.93, 1.29, 0.85, 0.68],
        'TxG_A': [0.84, 1.31, 1.51, 1.78, 1.41, 1.78, 1.64, 1.58, 2.05, 1.83, 2.14, 1.27, 1.50, 2.12, 2.25, 1.52, 2.48, 2.02, 2.94, 1.87]
    }
    return pd.DataFrame(bundesliga), pd.DataFrame(premier_league)

df_bl, df_epl = load_data()

# --- MODEL DIXON-COLES ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

def predict_match(df, h_team, a_team, w, rho):
    avg_h_gf = df['H_GF'].mean()
    avg_a_gf = df['T_GF'].mean()
    
    h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
    
    l_h_r = (h['HxG_F']*w[0] + h['H_GF']*w[1] + h['TxG_F']*w[2] + h['T_GF']*w[3])
    m_h_r = (h['HxG_A']*w[0] + h['H_GA']*w[1] + h['TxG_A']*w[2] + h['T_GA']*w[3])
    l_a_r = (a['TxG_F']*w[0] + a['T_GF']*w[1] + a['TxG_F']*w[2] + a['T_GF']*w[3])
    m_a_r = (a['TxG_A']*w[0] + a['T_GA']*w[1] + a['TxG_A']*w[2] + a['T_GA']*w[3])
    
    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)
    
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

def render_league(df, league_key, upcoming):
    st.sidebar.header(f"⚙️ Wagi {league_key}")
    if f'res_{league_key}' not in st.session_state: st.session_state[f'res_{league_key}'] = 0
    if st.sidebar.button(f"Reset {league_key} (45/30/15/10)"): st.session_state[f'res_{league_key}'] += 1
    
    opts = [i for i in range(0, 105, 5)]
    w0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", opts, index=9, key=f"w0_{league_key}_{st.session_state[f'res_{league_key}']}")
    w1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", opts, index=6, key=f"w1_{league_key}_{st.session_state[f'res_{league_key}']}")
    w2 = st.sidebar.selectbox("📊 xG Cały Sezon %", opts, index=3, key=f"w2_{league_key}_{st.session_state[f'res_{league_key}']}")
    w3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", opts, index=2, key=f"w3_{league_key}_{st.session_state[f'res_{league_key}']}")
    
    if (w0+w1+w2+w3) != 100: st.sidebar.error("Suma wag musi być 100%!"); st.stop()
    weights = [w0/100, w1/100, w2/100, w3/100]

    st.title(f"⚽ {league_key} Predictor Pro")
    c1, c2 = st.columns(2)
    with c1: h_sel = st.selectbox("Gospodarz", df['Team'], index=0, key=f"h_{league_key}")
    with c2: a_sel = st.selectbox("Gość", df['Team'], index=1, key=f"a_{league_key}")
    
    p1, px, p2, lf, mf, has, hds, aas, ads, mtx = predict_match(df, h_sel, a_sel, weights, 0.1)
    
    st.divider()
    m1, mx, m2 = st.columns(3)
    m1.metric(h_sel, f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
    mx.metric("Remis (Dixon-Coles)", f"{px:.1%}", f"Kurs: {1/px:.2f}")
    m2.metric(a_sel, f"{p2:.1%}", f"Kurs: {1/p2:.2f}")
    
    st.write("### 📊 Współczynniki Siły")
    def fmt(v, d=False):
        c = "green" if (v<1 if d else v>1) else "red"
        return f":{c}[{v:.2f} ({v-1:+.0%})]"
    st.markdown(f"| Drużyna | Atak | Obrona | Gole Exp. |\n| :--- | :--- | :--- | :--- |\n| **{h_sel}** | {fmt(has)} | {fmt(hds,True)} | **{lf:.2f}** |\n| **{a_sel}** | {fmt(aas)} | {fmt(ads,True)} | **{mf:.2f}** |")

    st.write("### ⚽ Macierz Prawdopodobieństwa")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mtx[:8, :8], annot=True, fmt=".1%", cmap="YlGn", cbar=False)
    plt.xlabel(f"Gole {a_sel}"); plt.ylabel(f"Gole {h_sel}")
    st.pyplot(fig)

    st.divider()
    st.write("### 🏦 Kalkulator Value Bet")
    vk1, vkx, vk2 = st.columns(3)
    with vk1: b1 = st.text_input(f"Kurs {h_sel}", "2.00", key=f"b1_{league_key}")
    with vkx: bx = st.text_input("Kurs X", "3.40", key=f"bx_{league_key}")
    with vk2: b2 = st.text_input(f"Kurs {a_sel}", "4.00", key=f"b2_{league_key}")

    def check(prob, bk):
        try:
            k = float(bk.replace(',', '.'))
            return f"✅ TAK ({k:.2f})" if k > (1/prob) else f"❌ NIE ({k:.2f})"
        except: return "-"

    st.table({"Typ": ["1", "X", "2"], "Model (%)": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"], "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"], "Value?": [check(p1, b1), check(px, bx), check(p2, b2)]})

    st.divider()
    st.subheader("📅 Terminarz i Kursy Fair")
    res = []
    for ht, at in upcoming:
        tp1, tpx, tp2, _, _, _, _, _, _, _ = predict_match(df, ht, at, weights, 0.1)
        res.append({"Mecz": f"{ht} [{1/tp1:.2f}] - {at} [{1/tp2:.2f}]", "Remis [X]": f"{1/tpx:.2f}"})
    st.table(pd.DataFrame(res))

    st.divider()
    st.subheader("📉 Analiza Under / Over")
    ou_cols = st.columns(4)
    for i, line in enumerate([1.5, 2.5, 3.5, 4.5]):
        u_p = sum(mtx[x, y] for x in range(12) for y in range(12) if x+y < line)
        with ou_cols[i]:
            st.write(f"**Linia {line}**")
            st.write(f"🟢 O: {1-u_p:.1%} (k: {1/(1-u_p):.2f})")
            st.write(f"🔴 U: {u_p:.1%} (k: {1/u_p:.2f})")
            st.progress(1-u_p)

with tab1:
    render_league(df_bl, "Bundesliga", [("Borussia Dortmund", "FC Cologne"), ("RB Leipzig", "Bayern Munich")])
with tab2:
    render_league(df_epl, "Premier League", [("Liverpool", "Everton"), ("Arsenal", "Manchester United")])
