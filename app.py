import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURACJA ---
st.set_page_config(page_title="Football Predictor PRO", layout="wide")

# --- POPRAWIONE DANE BAZOWE (PRZYKŁAD BAYERNU) ---
@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig'],
        'H_GF': [3.10, 2.33, 2.10, 2.25], # Poprawione realne średnie
        'H_GA': [0.80, 0.92, 0.95, 1.42],
        'HxG_F': [2.85, 2.00, 2.26, 2.65],
        'HxG_A': [0.90, 1.23, 0.92, 1.51],
        'T_GF': [2.95, 2.13, 2.05, 1.92],
        'T_GA': [0.90, 1.04, 1.10, 1.38],
        'Logo_ID': [27, 16, 15, 23826]
    }
    # Uproszczone dla czytelności przykładu, w pełnej wersji są wszystkie Twoje drużyny
    return pd.DataFrame(data)

# --- DANE Z TWOICH SCREENÓW (OSTATNIE 5-6 MECZÓW) ---
def get_recent_stats():
    return {
        'Bayern Munich': {
            'GF_last': 20/6,   # 3.33 gola/mecz (z Twojej tabeli Form)
            'GA_last': 8/6,    # 1.33 gola/mecz
            'xG_last': 16.10/5, # 3.22 xG/mecz (z Twojej tabeli xG)
            'xGA_last': 8.36/5  # 1.67 xGA/mecz
        },
        'Borussia Dortmund': {
            'GF_last': 15/6, 'GA_last': 9/6, 'xG_last': 8.22/5, 'xGA_last': 10.06/5
        }
    }

# --- LOGIKA OBLICZEŃ (TWOJA MATEMATYKA ZE SCREENÓW) ---
def calculate_detailed_bonus(team_name, base_xgf, base_xga):
    stats = get_recent_stats()
    if team_name not in stats: return 0, 0, {}
    
    s = stats[team_name]
    
    # 1. ANALIZA ATAKU
    # Trend Kreacji: (Aktualne xG / Bazowe xG) - 1
    trend_atk = (s['xG_last'] - base_xgf) / base_xgf
    # Skuteczność: (Gole strzelone / xG ostatnie) - 1
    skut_atk = (s['GF_last'] - s['xG_last']) / s['xG_last']
    # Bonus Ataku: (70% Trend + 30% Skuteczność) * 0.25 (tłumienie)
    bonus_atk = ((trend_atk * 0.7) + (skut_atk * 0.3)) * 0.25
    
    # 2. ANALIZA OBRONY
    # Trend Def: (Bazowe xGA / Aktualne xGA) - 1 (dodatnie jeśli dopuszczają mniej)
    trend_def = (base_xga - s['xGA_last']) / base_xga
    # Skuteczność Def: (xGA ostatnie / Gole stracone) - 1 (dodatnie jeśli tracą mniej niż xG)
    skut_def = (s['xGA_last'] - s['GA_last']) / s['xGA_last']
    # Bonus Obrony: (70% Trend + 30% Skuteczność) * 0.25
    bonus_def = ((trend_def * 0.7) + (skut_def * 0.3)) * 0.25
    
    details = {
        "base_xgf": base_xgf, "curr_xgf": s['xG_last'], "trend_atk": trend_atk,
        "curr_gf": s['GF_last'], "skut_atk": skut_atk, "bonus_atk": bonus_atk,
        "base_xga": base_xga, "curr_xga": s['xGA_last'], "trend_def": trend_def,
        "curr_ga": s['GA_last'], "skut_def": skut_def, "bonus_def": bonus_def
    }
    return bonus_atk, bonus_def, details

# --- UI ---
df = load_bundesliga()
st.title("⚽ Deep Dive Predictor: Bayern vs Dortmund")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)

h_row = df[df['Team'] == h_team].iloc[0]
a_row = df[df['Team'] == a_team].iloc[0]

# OBLICZENIA BONUSÓW
h_b_atk, h_b_def, h_logs = calculate_detailed_bonus(h_team, h_row['HxG_F'], h_row['HxG_A'])
a_b_atk, a_b_def, a_logs = calculate_detailed_bonus(a_team, a_row['HxG_F'], a_row['HxG_A'])

# --- SEKCJA MATEMATYCZNA (TO O CO PROSIŁEŚ) ---
st.header("🧮 Logika obliczeń krok po kroku")

def show_math_expander(team, logs):
    with st.expander(f"ZOBACZ JAK WYLICZONO BONUS DLA: {team}"):
        st.write("**1. Moduł Ataku:**")
        st.latex(rf"Trend = (\frac{{{logs['curr_xgf']:.2f}}}{{{logs['base_xgf']:.2f}}}) - 1 = {logs['trend_atk']:.2%}")
        st.latex(rf"Skuteczność = (\frac{{{logs['curr_gf']:.2f}}}{{{logs['curr_xgf']:.2f}}}) - 1 = {logs['skut_atk']:.2%}")
        st.info(f"Finalny Bonus Ataku: ({logs['trend_atk']:.2%}) * 0.7 + ({logs['skut_atk']:.2%}) * 0.3 = **{logs['bonus_atk']:.2%}**")
        
        st.write("**2. Moduł Obrony:**")
        st.latex(rf"Trend Def = (\frac{{{logs['base_xga']:.2f}}}{{{logs['curr_xga']:.2f}}}) - 1 = {logs['trend_def']:.2%}")
        st.latex(rf"Skuteczność Def = (\frac{{{logs['curr_xga']:.2f}}}{{{logs['curr_ga']:.2f}}}) - 1 = {logs['skut_def']:.2%}")
        st.info(f"Finalny Bonus Obrony: **{logs['bonus_def']:.2%}**")

show_math_expander(h_team, h_logs)
show_math_expander(a_team, a_logs)

# --- FINALNA LAMBDA POISSONA ---
# Lambda = (Bazowy Atak * Bazowa Obrona Przeciwnika) * (1 + Bonus Ataku - Bonus Obrony RYWALA)
avg_gf = df['H_GF'].mean()

# Uproszczona baza do przykładu
l_h_base = h_row['HxG_F'] 
l_a_base = a_row['HxG_F']

final_lambda = l_h_base * (1 + h_b_atk - a_b_def)
final_mu = l_a_base * (1 + a_b_atk - h_b_def)

st.divider()
st.subheader("🎯 Wynik Końcowy po uwzględnieniu trendów")
c1, c2 = st.columns(2)
c1.metric(f"ExG {h_team}", f"{final_lambda:.2f}", f"{h_b_atk:+.1%} (Forma)")
c2.metric(f"ExG {a_team}", f"{final_mu:.2f}", f"{a_b_atk:+.1%} (Forma)")

# --- MACIERZ ---
max_g = 7
matrix = np.zeros((max_g, max_g))
for x in range(max_g):
    for y in range(max_g):
        matrix[x, y] = poisson.pmf(x, final_lambda) * poisson.pmf(y, final_mu)

p1 = np.sum(np.tril(matrix, -1))
px = np.sum(np.diag(matrix))
p2 = np.sum(np.triu(matrix, 1))

st.write(f"**Prawdopodobieństwo: Dom: {p1:.1%} | Remis: {px:.1%} | Wyjazd: {p2:.1%}**")
