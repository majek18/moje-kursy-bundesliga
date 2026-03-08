import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor PRO", layout="wide", page_icon="⚽")

# --- DANE BAZOWE (PRZYKŁAD BUNDESLIGA) ---
@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen'],
        'H_GF': [4.00, 2.33, 2.25, 1.75, 2.25, 2.08],
        'H_GA': [1.00, 0.92, 1.17, 1.00, 1.42, 0.92],
        'T_GF': [3.67, 2.13, 2.04, 2.00, 1.92, 1.88],
        'T_GA': [0.96, 1.04, 1.29, 1.33, 1.38, 1.21],
        'HxG_F': [3.43, 2.00, 2.07, 2.11, 2.65, 2.26],
        'HxG_A': [1.04, 1.23, 1.28, 1.35, 1.51, 0.92],
        'TxG_F': [3.07, 1.85, 1.85, 1.96, 2.20, 2.02],
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62],
        'Logo_ID': [27, 16, 533, 79, 23826, 15]
    }
    return pd.DataFrame(data)

# --- FUNKCJE POMOCNICZE ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- RENDEROWANIE UI ---
def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    st.title(f"⚽ {league_name} Predictor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox("Gospodarz", df['Team'], index=0)
        h_f = st.select_slider(f"Forma {h_team} (Ostatnie 5 meczów)", options=list(range(-20, 21)), value=5)
    with col_b:
        a_team = st.selectbox("Gość", df['Team'], index=1)
        a_f = st.select_slider(f"Forma {a_team} (Ostatnie 5 meczów)", options=list(range(-20, 21)), value=-5)

    h = df[df['Team'] == h_team].iloc[0]
    a = df[df['Team'] == a_team].iloc[0]

    # --- PANEL OBLICZENIOWY ---
    st.markdown("### 🧮 Panel Obliczeniowy Modelu")
    with st.container(border=True):
        st.markdown("**1. Dane Wejściowe (Sezon vs Ostatnie 5 meczów)**")
        
        # Symulowane dane z ostatnich 5 meczów dla przykładu
        last_5_h = {"GF": h['H_GF']*1.2, "GA": h['H_GA']*1.1, "xG": h['HxG_F']*1.15, "xGA": h['HxG_A']*1.25}
        
        input_data = pd.DataFrame({
            "Parametr (na mecz)": ["Gole Strzelone (GF)", "Gole Stracone (GA)", "xG (Kreacja)", "xGA (Dopuszczone)"],
            "Sezon (Baza)": [f"{h['H_GF']:.2f}", f"{h['H_GA']:.2f}", f"{h['HxG_F']:.2f}", f"{h['HxG_A']:.2f}"],
            "Ostatnie 5 meczów": [f"{last_5_h['GF']:.2f}", f"{last_5_h['GA']:.2f}", f"{last_5_h['xG']:.2f}", f"{last_5_h['xGA']:.2f}"],
            "Analiza Trendu": ["🔥 Wzrost skuteczności", "⚠️ Defensywa przecieka", "📈 Kreują więcej", "📉 Rywale zagrażają"]
        })
        st.table(input_data)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**2. Obliczanie Bonusu Ataku ($B_{{atk}}$)**")
            t_kr = (last_5_h['xG'] - h['HxG_F']) / h['HxG_F']
            skut = (last_5_h['GF'] - last_5_h['xG']) / last_5_h['xG']
            sur_atk = (t_kr * 0.7) + (skut * 0.3)
            h_mod = h_f / 100 # Wartość z suwaka jako 'finalny' wynik tłumienia
            
            st.write(f"Trend Kreacji (70%): `{t_kr:+.1%}`")
            st.write(f"Skuteczność (30%): `{skut:+.1%}`")
            st.latex(rf"B_{{atk}} = ({t_kr:.3f} \cdot 0.7) + ({skut:.3f} \cdot 0.3) = {sur_atk:.2%}")
            st.write(f"Po tłumieniu (x 0.25): **{h_mod:+.2%}**")

        with c2:
            st.markdown(f"**3. Obliczanie Bonusu Obrony ($B_{{def}}$)**")
            t_def = (h['HxG_A'] - last_5_h['xGA']) / h['HxG_A']
            f_br = (last_5_h['xGA'] - last_5_h['GA']) / last_5_h['xGA']
            sur_def = (t_def * 0.7) + (f_br * 0.3)
            a_mod = a_f / 100
            
            st.write(f"Trend Defensywny (70%): `{t_def:+.1%}`")
            st.write(f"Forma Bramkarza (30%): `{f_br:+.1%}`")
            st.latex(rf"B_{{def}} = ({t_def:.3f} \cdot 0.7) + ({f_br:.3f} \cdot 0.3) = {sur_def:.2%}")
            st.write(f"Po tłumieniu (x 0.25): **{a_mod:+.2%}**")

        st.divider()
        st.markdown("**4. Finalne Parametry Poisson**")
        
        # Obliczenia wagowe (uproszczone do prezentacji)
        w_atk = (h['HxG_F'] * 0.6 + h['H_GF'] * 0.4) / avg_h_gf
        w_def = (a['TxG_A'] * 0.6 + a['A_GA'] * 0.4) / avg_h_gf
        l_base = w_atk * w_def * avg_h_gf
        
        # Gole Gości (mu)
        v_atk = (a['TxG_F'] * 0.6 + a['A_GF'] * 0.4) / avg_a_gf
        v_def = (h['HxG_A'] * 0.6 + h['H_GA'] * 0.4) / avg_a_gf
        m_base = v_atk * v_def * avg_a_gf

        l_final = l_base * (1 + h_mod)
        m_final = m_base * (1 + a_mod)

        badge_h = " ⚡" if h_mod != 0 else ""
        badge_a = " 🛡️" if a_mod != 0 else ""

        col_f1, col_f2 = st.columns(2)
        col_f1.latex(rf"\lambda_{{final}}{badge_h} = {l_base:.2f} \cdot (1 {h_mod:+.2f}) = \mathbf{{{l_final:.3f}}}")
        col_f2.latex(rf"\mu_{{final}}{badge_a} = {m_base:.2f} \cdot (1 {a_mod:+.2f}) = \mathbf{{{m_final:.3f}}}")

    # --- MACIERZ I WYNIKI ---
    max_g = 10
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            matrix[x,y] = (poisson.pmf(x, l_final) * poisson.pmf(y, m_final)) * dixon_coles_adjustment(x,y,l_final,m_final,-0.15)
    matrix /= matrix.sum()

    st.divider()
    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Wygrana {h_team}", f"{p1:.1%}")
    c2.metric("Remis", f"{px:.1%}")
    c3.metric(f"Wygrana {a_team}", f"{p2:.1%}")

# Start
tab1, tab2, tab3 = st.tabs(["Bundesliga", "Premier League", "La Liga"])
with tab1: render_league_ui(load_bundesliga(), "Bundesliga")
