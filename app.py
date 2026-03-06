import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# --- DANE Z TWOICH SCREENÓW (EPL + Bundesliga) ---
@st.cache_data
def load_data():
    bundesliga = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 'FC Cologne', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
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

# --- LOGIKA ZGODNA Z EXCELEM ---
def get_prediction_excel_style(df, h_team, a_team, w, rho):
    # Średnie z Twojego Excela
    avg_gf = 1.75
    avg_ga = 1.41
    
    h = df[df['Team'] == h_team].iloc[0]
    a = df[df['Team'] == a_team].iloc[0]
    
    # Sumy goli (Ważone)
    h_sum_f = (h['HxG_F']*w[0] + h['H_GF']*w[1] + h['TxG_F']*w[2] + h['T_GF']*w[3])
    h_sum_a = (h['HxG_A']*w[0] + h['H_GA']*w[1] + h['TxG_A']*w[2] + h['T_GA']*w[3])
    a_sum_f = (a['TxG_F']*w[0] + a['T_GF']*w[1] + a['TxG_F']*w[2] + a['T_GF']*w[3])
    a_sum_a = (a['TxG_A']*w[0] + a['T_GA']*w[1] + a['TxG_A']*w[2] + a['T_GA']*w[3])
    
    # Współczynniki siły
    h_atk, h_def = h_sum_f / avg_gf, h_sum_a / avg_ga
    a_atk, a_def = a_sum_f / avg_ga, a_sum_a / avg_gf
    
    # Surowa Lambda (Zgodna z Twoim 3,05 dla Bayernu)
    lambda_h = h_atk * a_def * avg_gf
    lambda_a = a_atk * h_def * avg_ga
    
    # Dixon-Coles Adjustment
    def dc_adj(x, y, lh, la, r):
        if x == 0 and y == 0: return 1 - (lh * la * r)
        if x == 0 and y == 1: return 1 + (lh * r)
        if x == 1 and y == 0: return 1 + (la * r)
        if x == 1 and y == 1: return 1 - r
        return 1

    matrix = np.zeros((12, 12))
    for x in range(12):
        for y in range(12):
            p = poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a)
            matrix[x, y] = p * dc_adj(x, y, lambda_h, lambda_a, rho)
    matrix /= matrix.sum()
    
    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    return p1, px, p2, lambda_h, lambda_a, h_atk, h_def, a_atk, a_def, matrix

# --- INTERFEJS ---
t1, t2 = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League"])

def render_ui(df, key, matches):
    st.sidebar.header(f"Konfiguracja {key}")
    w0 = st.sidebar.slider(f"xG Sezon D/W % ({key})", 0, 100, 45)
    w1 = st.sidebar.slider(f"Gole Sezon D/W % ({key})", 0, 100, 30)
    w2 = st.sidebar.slider(f"xG Cały Sezon % ({key})", 0, 100, 15)
    w3 = st.sidebar.slider(f"Gole Cały Sezon % ({key})", 0, 100, 10)
    
    if (w0+w1+w2+w3) != 100: st.sidebar.warning("Suma wag musi być 100%!")
    weights = [w0/100, w1/100, w2/100, w3/100]

    st.title(f"⚽ {key} Predictor (Excel Sync)")
    c1, c2 = st.columns(2)
    with c1: h_team = st.selectbox("Gospodarz", df['Team'], index=0, key=f"h_{key}")
    with c2: a_team = st.selectbox("Gość", df['Team'], index=11, key=f"a_{key}")

    p1, px, p2, lh, la, hatk, hdef, aatk, adef, mtx = get_prediction_excel_style(df, h_team, a_team, weights, 0.1)

    st.divider()
    res1, resx, res2 = st.columns(3)
    res1.metric(h_team, f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
    resx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
    res2.metric(a_team, f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

    # Współczynniki
    st.write("### 📊 Współczynniki (Zgodne z Excel)")
    st.markdown(f"| Drużyna | Atak | Obrona | Surowa Lambda |\n| :--- | :--- | :--- | :--- |\n| **{h_team}** | {hatk:.2f} | {hdef:.2f} | **{lh:.2f}** |\n| **{a_team}** | {aatk:.2f} | {adef:.2f} | **{la:.2f}** |")

    # Macierz
    st.write("### ⚽ Macierz wyników")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mtx[:6, :6], annot=True, fmt=".1%", cmap="Blues", cbar=False)
    plt.xlabel(a_team); plt.ylabel(h_team)
    st.pyplot(fig)

    # Value Bet
    st.divider()
    st.write("### 🏦 Kalkulator Value Bet")
    bk1, bkx, bk2 = st.columns(3)
    k1 = bk1.text_input(f"Bukmacherski {h_team}", "2.10", key=f"k1_{key}")
    kx = bkx.text_input("Bukmacherski X", "3.50", key=f"kx_{key}")
    k2 = bk2.text_input(f"Bukmacherski {a_team}", "4.20", key=f"k2_{key}")

    def is_val(p, k):
        try:
            kv = float(k.replace(',','.'))
            return f"✅ TAK ({kv:.2f})" if kv > (1/p) else f"❌ NIE ({kv:.2f})"
        except: return "-"

    st.table({"Typ": ["1", "X", "2"], "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"], "Value?": [is_val(p1, k1), is_val(px, kx), is_val(p2, k2)]})

    # Terminarz
    st.divider()
    st.subheader("📅 Terminarz")
    sched = []
    for ht, at in matches:
        tp1, tpx, tp2, _, _, _, _, _, _, _ = get_prediction_excel_style(df, ht, at, weights, 0.1)
        sched.append({"Mecz": f"{ht} [{1/tp1:.2f}] - {at} [{1/tp2:.2f}]", "Remis": f"{1/tpx:.2f}"})
    st.table(pd.DataFrame(sched))

    # Under Over
    st.divider()
    st.subheader("📉 Under / Over")
    cols = st.columns(4)
    for i, line in enumerate([1.5, 2.5, 3.5, 4.5]):
        up = sum(mtx[x,y] for x in range(12) for y in range(12) if x+y < line)
        with cols[i]:
            st.write(f"**Linia {line}**")
            st.write(f"🟢 O: {1-up:.1%} ({1/(1-up):.2f})")
            st.write(f"🔴 U: {up:.1%} ({1/up:.2f})")
            st.progress(1-up)

with t1:
    render_ui(df_bl, "Bundesliga", [("Bayern Munich", "Borussia M.Gladbach"), ("RB Leipzig", "Hoffenheim")])
with t2:
    render_ui(df_epl, "Premier League", [("Arsenal", "Everton"), ("Manchester City", "Chelsea")])
