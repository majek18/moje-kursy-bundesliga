import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
import requests

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor PRO", layout="wide", page_icon="⚽")

# --- DANE ZESKANOWANE ZE SCREENÓW (MARZEC 2026) ---
@st.cache_data
def load_current_form_data():
    form_data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Augsburg', 'VfB Stuttgart', 'RB Leipzig', 
                 'Hoffenheim', 'Hamburger SV', 'St. Pauli', 'Bayer Leverkusen', 'Mainz 05', 
                 'Eintracht Frankfurt', 'Freiburg', 'Borussia M.Gladbach', 'FC Koln', 
                 'Werder Bremen', 'Union Berlin', 'FC Heidenheim', 'Wolfsburg'],
        'Pts': [16, 13, 12, 11, 11, 10, 10, 10, 9, 9, 7, 7, 5, 4, 4, 4, 1, 1],
        # Dane do obliczeń bonusów (Sezon vs Ostatnie)
        'S_GF': [4.00, 1.90, 1.50, 2.30, 1.80, 2.10, 1.50, 1.10, 1.60, 1.30, 1.60, 1.00, 0.80, 1.20, 0.60, 0.80, 1.10, 1.00],
        'R_GF': [4.80, 1.20, 1.40, 2.10, 1.90, 2.20, 1.30, 1.20, 1.70, 1.50, 1.40, 0.90, 1.10, 1.00, 0.70, 0.90, 1.00, 0.90],
        'S_GA': [1.00, 1.40, 1.10, 1.30, 1.50, 1.80, 1.10, 1.50, 1.00, 1.50, 1.60, 1.60, 1.80, 1.80, 1.50, 1.80, 2.50, 2.30],
        'R_GA': [1.10, 1.00, 1.20, 1.40, 1.20, 1.90, 1.20, 1.40, 1.10, 1.40, 1.50, 1.70, 1.90, 1.90, 1.60, 1.90, 2.60, 2.40],
        'S_xG': [3.43, 1.85, 1.60, 2.10, 2.20, 1.90, 1.40, 1.00, 1.80, 1.60, 1.50, 1.10, 1.20, 1.30, 0.90, 1.00, 1.30, 1.20],
        'R_xG': [3.94, 1.70, 1.70, 2.00, 2.30, 2.00, 1.30, 0.90, 1.90, 1.80, 1.40, 1.20, 1.30, 1.20, 1.00, 1.10, 1.20, 1.10],
        'S_xGA': [1.04, 1.50, 1.30, 1.40, 1.40, 1.70, 1.20, 1.60, 1.10, 1.70, 1.70, 1.70, 1.90, 1.90, 1.60, 1.70, 2.40, 2.10],
        'R_xGA': [1.30, 1.10, 1.40, 1.50, 1.30, 1.80, 1.30, 1.50, 1.20, 1.60, 1.60, 1.80, 2.00, 1.80, 1.70, 1.80, 2.50, 2.20],
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(form_data)

# --- FUNKCJE POMOCNICZE ---
def calculate_team_bonuses(team_name, form_df):
    t = form_df[form_df['Team'] == team_name].iloc[0]
    
    # Atak
    trend_k_val = (t['R_xG'] - t['S_xG']) / t['S_xG']
    skut_val = (t['R_GF'] - t['R_xG']) / t['R_xG']
    raw_atk = (trend_k_val * 0.7) + (skut_val * 0.3)
    final_atk = raw_atk * 0.25 # Tłumienie
    
    # Obrona
    trend_d_val = (t['S_xGA'] - t['R_xGA']) / t['S_xGA']
    skut_d_val = (t['R_xGA'] - t['R_GA']) / t['R_xGA']
    raw_def = (trend_d_val * 0.7) + (skut_d_val * 0.3)
    final_def = raw_def * 0.25 # Tłumienie
    
    return {
        "atk": final_atk, "def": final_def, "raw_atk": raw_atk, "raw_def": raw_def,
        "trend_k": trend_k_val, "skut": skut_val, "trend_d": trend_d_val, "skut_d": skut_d_val
    }

def get_trend_icon(val, is_defense=False):
    if is_defense:
        return "🛡️ Poprawa w defensywie" if val < 0 else "⚠️ Defensywa przecieka"
    if val > 0.05: return "🔥 Wzrost skuteczności"
    if val < -0.05: return "🧊 Kryzys skuteczności"
    return "📊 Stabilna forma"

# --- [DANE BAZOWE BUNDESLIGA, PREMIER LEAGUE, LA LIGA - BEZ ZMIAN] ---
@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Hoffenheim', 'VfB Stuttgart', 'RB Leipzig', 'Bayer Leverkusen', 
                 'Eintracht Frankfurt', 'Freiburg', 'Augsburg', 'Union Berlin', 'Hamburger SV', 'Borussia M.Gladbach', 
                 'FC Koln', 'Mainz 05', 'St. Pauli', 'Werder Bremen', 'Wolfsburg', 'FC Heidenheim'],
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
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

# --- SIDEBAR WAGI ---
st.sidebar.header("⚙️ Konfiguracja Wag")
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", [i for i in range(0, 105, 5)], index=8)
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", [i for i in range(0, 105, 5)], index=5)
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", [i for i in range(0, 105, 5)], index=4)
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", [i for i in range(0, 105, 5)], index=3)
w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100

# --- MODUŁ LIGI ---
def render_league_ui(df, league_name):
    form_df = load_current_form_data()
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=1, key=f"a_{league_name}")

    # --- NOWY MODUŁ: ANALIZA FORMY ---
    st.divider()
    st.subheader(f"📈 Analiza Formy: {h_team} vs {a_team}")
    
    res_h = calculate_team_bonuses(h_team, form_df)
    res_a = calculate_team_bonuses(a_team, form_df)
    
    for team, res, label in [(h_team, res_h, "Gospodarz"), (a_team, res_a, "Gość")]:
        t = form_df[form_df['Team'] == team].iloc[0]
        st.markdown(f"#### Dane Wejściowe dla {team}")
        
        # Tabela 1: Dane Wejściowe
        input_data = {
            "Parametr (na mecz)": ["Gole Strzelone (GF)", "Gole Stracone (GA)", "xG (Kreacja)", "xGA (Dopuszczone)"],
            "Sezon (Baza)": [f"{t['S_GF']:.2f}", f"{t['S_GA']:.2f}", f"{t['S_xG']:.2f}", f"{t['S_xGA']:.2f}"],
            "Ostatnie 5 meczów": [f"{t['R_GF']:.2f}", f"{t['R_GA']:.2f}", f"{t['R_xG']:.2f}", f"{t['R_xGA']:.2f}"],
            "Analiza Trendu": [
                get_trend_icon(t['R_GF'] - t['S_GF']), 
                get_trend_icon(t['R_GA'] - t['S_GA'], True),
                "📈 Wzrost szans" if res['trend_k'] > 0 else "📉 Spadek szans",
                "✅ Solidna obrona" if res['trend_d'] > 0 else "❌ Słaba obrona"
            ]
        }
        st.table(pd.DataFrame(input_data))

        # Obliczenia Bonusów
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Obliczanie Bonusu Ataku ($B_{{atk\_H if label=='Gospodarz' else 'A'}}$)**")
            st.write(f"Trend Kreacji (70%): **{res['trend_k']:+.1%}**")
            st.write(f"Skuteczność (30%): **{res['skut']:+.1%}**")
            st.latex(rf"B_{{atk}} = ({res['raw_atk']:.2%}) \times 0.25 = {res['atk']:+.2%}")
        with c2:
            st.markdown(f"**Obliczanie Bonusu Obrony ($B_{{def\_H if label=='Gospodarz' else 'A'}}$)**")
            st.write(f"Trend Defensywny (70%): **{res['trend_d']:+.1%}**")
            st.write(f"Skuteczność Obrony/GK (30%): **{res['skut_d']:+.1%}**")
            st.latex(rf"B_{{def}} = ({res['raw_def']:.2%}) \times 0.25 = {res['def']:+.2%}")
        st.divider()

    # --- OBLICZENIA FINALNE ---
    h_data, a_data = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
    
    # Bazowe R
    l_h_r = (h_data['HxG_F']*w0 + h_data['H_GF']*w1 + h_data['AxG_F']*w2 + h_data['T_GF']*w3)
    m_h_r = (h_data['HxG_A']*w0 + h_data['H_GA']*w1 + h_data['AxG_A']*w2 + h_data['T_GA']*w3)
    l_a_r = (a_data['TxG_F']*w0 + a_data['A_GF']*w1 + a_data['AxG_F']*w2 + a_data['T_GF']*w3)
    m_a_r = (a_data['TxG_A']*w0 + a_data['A_GA']*w1 + a_data['AxG_A']*w2 + a_data['T_GA']*w3)

    # Finalne Lambdy z bonusami
    lambda_f = (l_h_r / avg_h_gf) * (m_a_r / avg_h_gf) * avg_h_gf * (1 + res_h['atk']) * (1 - res_a['def'])
    mu_f = (l_a_r / avg_a_gf) * (m_h_r / avg_a_gf) * avg_a_gf * (1 + res_a['atk']) * (1 - res_h['def'])

    st.subheader("4. Finalne starcie (Parametry Poissona)")
    st.latex(rf"\lambda_{{final}} = {lambda_f:.3f} \quad | \quad \mu_{{final}} = {mu_f:.3f}")

    # [Tutaj pozostała część Twojego kodu: Macierz, Symulacja, Value Bet Detector]
    # ... (kod identyczny jak w Twoim szablonie)

with st.tabs(["🇩🇪 Bundesliga"])[0]: render_league_ui(load_bundesliga(), "Bundesliga")
