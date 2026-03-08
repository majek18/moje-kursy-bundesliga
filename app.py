import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
import requests

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor", layout="wide", page_icon="⚽")

# --- MAPOWANIE LIG DLA API ---
LEAGUE_MAP = {
    "Bundesliga": "soccer_germany_bundesliga",
    "Premier League": "soccer_upcoming",
    "La Liga": "soccer_spain_la_liga"
}

# --- FUNKCJA POBIERANIA KURSÓW ---
def get_live_odds(league_key):
    try:
        api_key = st.secrets["ODDS_API_KEY"]
        url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# --- DANE BAZOWE: BUNDESLIGA ---
@st.cache_data
def load_bundesliga():
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
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.63, 1.90, 1.83, 1.72, 1.96, 2.22],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.06, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38],
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

# --- DANE BAZOWE: PREMIER LEAGUE ---
@st.cache_data
def load_premier_league():
    data = {
        'Team': ['Arsenal', 'Manchester City', 'Manchester United', 'Aston Villa', 'Chelsea', 'Liverpool', 'Brentford', 'Everton', 'Bournemouth', 'Fulham', 'Sunderland', 'Newcastle', 'Crystal Palace', 'Brighton', 'Leeds', 'Tottenham', 'Nottingham Forest', 'West Ham', 'Burnley', 'Wolves'],
        'H_GF': [2.35, 2.40, 1.92, 1.40, 1.64, 1.85, 1.71, 1.20, 1.40, 1.60, 1.57, 1.86, 1.00, 1.46, 1.46, 1.20, 0.92, 1.21, 1.07, 1.06],
        'H_GA': [0.64, 0.73, 1.14, 1.00, 1.14, 1.14, 1.07, 1.26, 1.00, 1.20, 0.92, 1.60, 1.28, 1.06, 1.33, 1.66, 1.35, 1.92, 1.64, 1.93],
        'T_GF': [1.96, 2.03, 1.75, 1.34, 1.82, 1.65, 1.51, 1.17, 1.51, 1.37, 1.03, 1.44, 1.13, 1.31, 1.27, 1.34, 0.96, 1.20, 1.10, 0.73],
        'T_GA': [0.73, 0.93, 1.37, 1.17, 1.17, 1.34, 1.37, 1.13, 1.58, 1.48, 1.13, 1.48, 1.20, 1.24, 1.65, 1.58, 1.48, 1.86, 2.00, 1.73],
        'HxG_F': [2.05, 2.23, 2.13, 1.36, 2.14, 1.90, 2.07, 1.36, 1.63, 1.39, 1.17, 2.19, 1.94, 1.41, 1.76, 1.24, 1.54, 1.39, 1.03, 1.14],
        'HxG_A': [0.74, 1.07, 1.01, 1.32, 1.54, 1.06, 1.31, 1.44, 0.75, 1.35, 1.46, 1.45, 1.51, 1.31, 1.32, 1.58, 1.59, 1.66, 1.88, 1.73],
        'TxG_F': [1.96, 2.01, 1.91, 1.34, 2.12, 1.86, 1.76, 1.30, 1.71, 1.26, 1.00, 1.63, 1.67, 1.45, 1.51, 1.18, 1.20, 1.29, 0.94, 0.93],
        'TxG_A': [0.79, 1.19, 1.27, 1.54, 1.47, 1.27, 1.47, 1.51, 1.45, 1.58, 1.61, 1.37, 1.50, 1.47, 1.54, 1.55, 1.72, 1.84, 2.16, 1.74],
        'A_GF': [1.62, 1.64, 1.60, 1.28, 2.00, 1.46, 1.33, 1.14, 1.64, 1.14, 0.53, 1.00, 1.26, 1.14, 1.00, 1.50, 1.00, 1.20, 1.13, 0.35],
        'A_GA': [0.81, 1.14, 1.60, 1.35, 1.20, 1.53, 1.66, 1.00, 2.21, 1.78, 1.40, 1.35, 1.13, 1.42, 2.00, 1.50, 1.83, 2.08, 2.33, 1.50],
        'AxG_F': [1.87, 1.78, 1.70, 1.32, 2.10, 1.81, 1.48, 1.22, 1.79, 1.11, 0.91, 1.03, 1.43, 1.48, 1.23, 1.10, 0.90, 1.20, 0.85, 0.68],
        'AxG_A': [0.84, 1.31, 1.51, 1.78, 1.41, 1.47, 1.62, 1.59, 2.20, 1.83, 1.75, 1.27, 1.49, 1.64, 1.77, 1.53, 1.85, 2.04, 2.43, 1.75],
        'Logo_ID': [11, 281, 985, 405, 631, 31, 1148, 29, 1003, 931, 289, 762, 873, 1237, 399, 148, 703, 379, 1132, 543]
    }
    return pd.DataFrame(data)

# --- DANE BAZOWE: LA LIGA ---
@st.cache_data
def load_la_liga():
    data = {
        'Team': ['Barcelona', 'Real Madrid', 'Atletico Madrid', 'Villarreal', 'Real Betis', 'Celta Vigo', 'Espanyol', 'Real Sociedad', 'Athletic Club', 'Osasuna', 'Getafe', 'Girona', 'Rayo Vallecano', 'Sevilla', 'Valencia', 'Alaves', 'Elche', 'Mallorca', 'Levante', 'Real Oviedo'],
        'H_GF': [3.15, 2.23, 2.36, 2.23, 1.92, 1.43, 1.23, 1.85, 1.21, 1.85, 0.77, 1.00, 1.15, 1.38, 1.23, 1.23, 1.57, 1.46, 1.00, 0.38],
        'H_GA': [0.46, 0.69, 0.86, 0.85, 1.15, 1.21, 1.38, 1.54, 1.14, 1.23, 0.85, 1.62, 0.77, 1.46, 1.00, 1.15, 1.07, 1.31, 1.71, 1.08],
        'A_GF': [2.21, 1.93, 1.00, 1.46, 1.31, 1.31, 1.31, 1.14, 1.00, 0.57, 0.85, 1.07, 0.85, 1.23, 0.85, 0.54, 1.00, 0.86, 1.15, 0.85],
        'A_GA': [1.43, 1.00, 1.00, 1.54, 1.31, 1.00, 1.62, 1.50, 1.62, 1.14, 1.38, 1.57, 1.69, 1.69, 2.00, 1.46, 2.00, 1.93, 1.62, 2.23],
        'T_GF': [2.67, 2.07, 1.70, 1.85, 1.62, 1.37, 1.27, 1.48, 1.11, 1.19, 0.81, 1.04, 1.00, 1.31, 1.04, 0.88, 1.31, 1.15, 1.07, 0.62],
        'T_GA': [0.96, 0.85, 0.93, 1.19, 1.23, 1.11, 1.50, 1.52, 1.37, 1.19, 1.12, 1.59, 1.23, 1.58, 1.50, 1.31, 1.50, 1.63, 1.67, 1.65],
        'HxG_F': [2.92, 2.72, 2.46, 2.04, 1.97, 1.49, 1.63, 1.85, 1.71, 1.72, 0.75, 1.35, 1.74, 1.16, 1.63, 1.65, 1.75, 1.44, 1.59, 0.96],
        'HxG_A': [0.90, 0.94, 0.87, 1.27, 1.26, 1.28, 1.59, 1.51, 0.81, 1.15, 0.98, 2.00, 1.03, 1.63, 0.95, 1.37, 1.85, 1.38, 2.06, 1.47],
        'TxG_F': [2.84, 2.01, 1.11, 1.55, 1.44, 1.35, 1.35, 1.24, 1.34, 0.91, 0.91, 1.40, 1.28, 1.02, 1.06, 1.20, 0.67, 0.85, 1.36, 1.15],
        'TxG_A': [1.89, 1.41, 1.47, 1.61, 1.38, 1.42, 1.63, 1.57, 1.57, 1.59, 1.49, 1.63, 1.95, 1.91, 2.02, 1.53, 2.08, 2.25, 1.93, 2.15],
        'AxG_F': [2.88, 2.35, 1.81, 1.80, 1.71, 1.42, 1.49, 1.53, 1.53, 1.30, 0.83, 1.38, 1.51, 1.09, 1.35, 1.43, 1.25, 1.13, 1.42, 1.05],
        'AxG_A': [1.41, 1.18, 1.16, 1.44, 1.32, 1.35, 1.61, 1.54, 1.18, 1.38, 1.24, 1.81, 1.49, 1.77, 1.48, 1.45, 1.96, 1.83, 1.92, 1.81],
        'Logo_ID': [131, 418, 13, 1050, 150, 940, 714, 681, 621, 331, 3709, 12321, 371, 33, 123, 1108, 1531, 237, 335, 338]
    }
    return pd.DataFrame(data)

# --- FUNKCJA KOREKTY ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SESSION STATE ---
if 'mod_reset' not in st.session_state: st.session_state.mod_reset = 0
def reset_mods(): st.session_state.mod_reset += 1

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfiguracja Wag")
if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0
def reset_weights(): st.session_state.reset_counter += 1
st.sidebar.button("🔄 Resetuj wagi", on_click=reset_weights)

options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", options, index=options.index(40), key=f"w0_{st.session_state.reset_counter}")
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", options, index=options.index(25), key=f"w1_{st.session_state.reset_counter}")
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", options, index=options.index(20), key=f"w2_{st.session_state.reset_counter}")
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, options.index(15), key=f"w3_{st.session_state.reset_counter}")

if v0 + v1 + v2 + v3 != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100
fixed_rho = -0.15

# --- INTERFEJS ---
tab_bl, tab_pl, tab_ll = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "🇪🇸 La Liga"])

def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    st.title(f"⚽ {league_name} Predictor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
        h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=100)
        with st.expander("🛠️ Modyfikatory Gospodarza"):
            mod_range = list(range(-20, 21))
            m_key = st.session_state.mod_reset
            h_k = st.select_slider("KONTUZJE", options=mod_range, value=0, key=f"h_k_{league_name}_{m_key}")
            h_f = st.select_slider("FORMA", options=mod_range, value=0, key=f"h_f_{league_name}_{m_key}")
            h_s = st.select_slider("STYL GRY", options=mod_range, value=0, key=f"h_s_{league_name}_{m_key}")
            h_p = st.select_slider("POGODA", options=mod_range, value=0, key=f"h_p_{league_name}_{m_key}")
            h_total_mod = (h_k + h_f + h_s + h_p) / 100
            st.button("🧹 Resetuj", key=f"reset_h_{league_name}", on_click=reset_mods, use_container_width=True)
            
    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=1, key=f"a_{league_name}")
        a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=100)
        with st.expander("🛠️ Modyfikatory Gościa"):
            mod_range = list(range(-20, 21))
            m_key = st.session_state.mod_reset
            a_k = st.select_slider("KONTUZJE", options=mod_range, value=0, key=f"a_k_{league_name}_{m_key}")
            a_f = st.select_slider("FORMA", options=mod_range, value=0, key=f"a_f_{league_name}_{m_key}")
            a_s = st.select_slider("STYL GRY", options=mod_range, value=0, key=f"a_s_{league_name}_{m_key}")
            a_p = st.select_slider("POGODA", options=mod_range, value=0, key=f"a_p_{league_name}_{m_key}")
            a_total_mod = (a_k + a_f + a_s + a_p) / 100
            st.button("🧹 Resetuj", key=f"reset_a_{league_name}", on_click=reset_mods, use_container_width=True)

    # --- NOWA SEKCJA: BONUS FORMY ---
    st.divider()
    st.markdown("### 📈 Bonus Formy (Dane z ostatniego okresu)")
    
    def calculate_form_bonus(team_name, base_gf, base_ga, base_xf, base_xa, key_prefix):
        st.markdown(f"**Dane dla {team_name}**")
        c1, c2, c3, c4 = st.columns(4)
        last_gf = c1.number_input("Gole Strzelone", value=float(base_gf), step=0.1, key=f"{key_prefix}_gf")
        last_ga = c2.number_input("Gole Stracone", value=float(base_ga), step=0.1, key=f"{key_prefix}_ga")
        last_xf = c3.number_input("xG (Kreacja)", value=float(base_xf), step=0.1, key=f"{key_prefix}_xf")
        last_xa = c4.number_input("xGA (Dopuszczone)", value=float(base_xa), step=0.1, key=f"{key_prefix}_xa")
        
        # Trend Kreacji (70%)
        trend_kreacji = (last_xf - base_xf) / max(base_xf, 0.1)
        # Skuteczność (30%)
        skutecznosc = (last_gf - last_xf) / max(last_xf, 0.1)
        # Surowy bonus ataku
        raw_atk = (trend_kreacji * 0.7) + (skutecznosc * 0.3)
        # Bonus po tłumieniu (x0.25)
        final_atk_bonus = raw_atk * 0.25
        
        # Trend Defensywny (70%)
        trend_def = (base_xa - last_xa) / max(base_xa, 0.1)
        # Forma Bramkarza (30%)
        forma_gk = (last_xa - last_ga) / max(last_xa, 0.1)
        # Surowy bonus obrony
        raw_def = (trend_def * 0.7) + (forma_gk * 0.3)
        # Bonus po tłumieniu (x0.25)
        final_def_bonus = raw_def * 0.25
        
        return final_atk_bonus, final_def_bonus

    h_data_row = df[df['Team'] == h_team].iloc[0]
    a_data_row = df[df['Team'] == a_team].iloc[0]

    col_form_h, col_form_a = st.columns(2)
    with col_form_h:
        h_bonus_atk, h_bonus_def = calculate_form_bonus(h_team, h_data_row['H_GF'], h_data_row['H_GA'], h_data_row['HxG_F'], h_data_row['HxG_A'], f"h_form_{league_name}")
        st.info(f"Bonus Atak: {h_bonus_atk:+.1%} | Bonus Obrona: {h_bonus_def:+.1%}")
        
    with col_form_a:
        a_bonus_atk, a_bonus_def = calculate_form_bonus(a_team, a_data_row['A_GF'], a_data_row['A_GA'], a_data_row['AxG_F'], a_data_row['AxG_A'], f"a_form_{league_name}")
        st.info(f"Bonus Atak: {a_bonus_atk:+.1%} | Bonus Obrona: {a_bonus_def:+.1%}")

    # --- OBLICZENIA POISSONA ---
    h, a = h_data_row, a_data_row
    l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['AxG_F']*w2 + h['T_GF']*w3)
    m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['AxG_A']*w2 + h['T_GA']*w3)
    l_a_r = (a['TxG_F']*w0 + a['A_GF']*w1 + a['AxG_F']*w2 + a['T_GF']*w3)
    m_a_r = (a['TxG_A']*w0 + a['A_GA']*w1 + a['AxG_A']*w2 + a['T_GA']*w3)

    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)

    # Aplikacja Bonusów Formy do Lambdy
    lambda_f = (h_atk_s * a_def_s * avg_h_gf) * (1 + h_total_mod + h_bonus_atk + a_bonus_def)
    mu_f = (a_atk_s * h_def_s * avg_a_gf) * (1 + a_total_mod + a_bonus_atk + h_bonus_def)

    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, fixed_rho)
    matrix /= matrix.sum()

    p1, px, p2 = np.sum(np.tril(matrix, -1).T), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1).T)
    model_odds = [1/max(p1, 0.001), 1/max(px, 0.001), 1/max(p2, 0.001)]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {model_odds[0]:.2f}")
    c2.metric("Remis", f"{px:.1%}", f"Kurs: {model_odds[1]:.2f}")
    c3.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {model_odds[2]:.2f}")

    st.markdown("#### ⚽ Przewidywana liczba goli (ExG)")
    ex_h, ex_a = st.columns(2)
    ex_h.metric(f"ExG {h_team}", f"{lambda_f:.2f}")
    ex_a.metric(f"ExG {a_team}", f"{mu_f:.2f}")

    st.divider()
    st.subheader("📊 Porównanie statystyk ze średnią ligową")
    
    def color_stat(val, avg, is_defense=False):
        if not is_defense:
            color = "#28a745" if val >= avg else "#dc3545"
        else:
            color = "#28a745" if val <= avg else "#dc3545"
        return f'background-color: {color}; color: white; font-weight: bold'

    def create_stat_styled_table(team_data, context, full_df):
        if context == "Cały sezon":
            gf, ga, xgf, xga = team_data['T_GF'], team_data['T_GA'], team_data['AxG_F'], team_data['AxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['T_GF'].mean(), full_df['T_GA'].mean(), full_df['AxG_F'].mean(), full_df['AxG_A'].mean()
        elif context == "Dom":
            gf, ga, xgf, xga = team_data['H_GF'], team_data['H_GA'], team_data['HxG_F'], team_data['HxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['H_GF'].mean(), full_df['H_GA'].mean(), full_df['HxG_F'].mean(), full_df['HxG_A'].mean()
        else:
            gf, ga, xgf, xga = team_data['A_GF'], team_data['A_GA'], team_data['TxG_F'], team_data['TxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['A_GF'].mean(), full_df['A_GA'].mean(), full_df['TxG_F'].mean(), full_df['TxG_A'].mean()

        df_stats = pd.DataFrame({
            "Statystyka": ["Gole Strzelone", "Gole Stracone", "xG (Atak)", "xG (Obrona)"],
            "Drużyna": [gf, ga, xgf, xga],
            "Średnia ligi": [l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga]
        })
        def apply_styling(row):
            is_def = "Stracone" in row["Statystyka"] or "Obrona" in row["Statystyka"]
            style = color_stat(row["Drużyna"], row["Średnia ligi"], is_def)
            return [None, style, None]
        return df_stats.style.apply(apply_styling, axis=1).format("{:.2f}", subset=["Drużyna", "Średnia ligi"])

    col_stats_h, col_stats_a = st.columns(2)
    with col_stats_h:
        st.markdown(f"**Zakres dla {h_team}**")
        ctx_h = st.radio("Wybierz:", ["Cały sezon", "Dom", "Wyjazd"], horizontal=True, key=f"ctx_h_{league_name}")
        st.table(create_stat_styled_table(h, ctx_h, df))
    with col_stats_a:
        st.markdown(f"**Zakres dla {a_team}**")
        ctx_a = st.radio("Wybierz:", ["Cały sezon", "Dom", "Wyjazd"], horizontal=True, key=f"ctx_a_{league_name}")
        st.table(create_stat_styled_table(a, ctx_a, df))

    st.divider()
    st.markdown("### 📊 Porównanie Siły Zespołów")
    def format_strength(val, is_attack=True):
        pct = (val - 1.0) * 100
        color = "green" if (is_attack and val >= 1) or (not is_attack and val <= 1) else "red"
        return f":{color}[{val:.2f} ({pct:+.0f}%)]"

    st.markdown(f"""
    | Cecha | {h_team} (Gospodarz) | {a_team} (Gość) |
    | :--- | :--- | :--- |
    | **Siła Ataku** | {format_strength(h_atk_s, True)} | {format_strength(a_atk_s, True)} |
    | **Siła Obrony** | {format_strength(h_def_s, False)} | {format_strength(a_def_s, False)} |
    | **Bonus Formy (A/O)** | **{h_bonus_atk:+.1%} / {h_bonus_def:+.1%}** | **{a_bonus_atk:+.1%} / {a_bonus_def:+.1%}** |
    | **Łączny Mod. Zewnętrzny** | **{h_total_mod:+.0%}** | **{a_total_mod:+.0%}** |
    """)

    with st.expander("🧮 Szczegółowa Ścieżka Obliczeniowa"):
        st.subheader("1. Średnie ligowe")
        st.write(f"Średnia gospodarzy: `{avg_h_gf:.3f}` | Średnia gości: `{avg_a_gf:.3f}`")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**{h_team}**")
            st.write(f"🎯 **Bazowa Siła Ataku:** `{l_h_r:.3f} / {avg_h_gf:.3f} = {h_atk_s:.3f}`")
        with sc2:
            st.markdown(f"**{a_team}**")
            st.write(f"🎯 **Bazowa Siła Ataku:** `{l_a_r:.3f} / {avg_a_gf:.3f} = {a_atk_s:.3f}`")
        st.subheader("2. Parametry Poisson (Skorygowane o Mod + Formę)")
        st.latex(rf"\lambda_{{final}} = \lambda_{{base}} \times (1 + {h_total_mod + h_bonus_atk + a_bonus_def:+.2f}) = {lambda_f:.3f}")
        st.latex(rf"\mu_{{final}} = \mu_{{base}} \times (1 + {a_total_mod + a_bonus_atk + h_bonus_def:+.2f}) = {mu_f:.3f}")

    with st.expander("📊 Zobacz Macierz Prawdopodobieństwa"):
        limit = 8
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".1%", cmap="YlGn", cbar=False)
        plt.xlabel(f"Gole {a_team}") 
        plt.ylabel(f"Gole {h_team}") 
        st.pyplot(fig)

    st.divider()
    st.subheader("📉 Analiza Under / Over")
    lines = [1.5, 2.5, 3.5, 4.5]
    ou_cols = st.columns(len(lines))
    for i, line in enumerate(lines):
        prob_under = sum(matrix[x, y] for x in range(max_g) for y in range(max_g) if x + y < line)
        prob_over = 1 - prob_under
        with ou_cols[i]:
            st.markdown(f"**Linia {line}**")
            st.write(f"🟢 **OVER**: {prob_over:.1%} (Kurs: {1/max(prob_over, 0.001):.2f})")
            st.write(f"🔴 **UNDER**: {prob_under:.1%} (Kurs: {1/max(prob_under, 0.001):.2f})")

    st.divider()
    st.subheader("🥅 Obie Drużyny Strzelą (BTTS)")
    prob_btts_yes = sum(matrix[x, y] for x in range(1, max_g) for y in range(1, max_g))
    prob_btts_no = 1 - prob_btts_yes
    b1, b2 = st.columns(2)
    with b1:
        st.write(f"🟢 **TAK**: {prob_btts_yes:.1%} (Kurs: {1/max(prob_btts_yes, 0.001):.2f})")
    with b2:
        st.write(f"🔴 **NIE**: {prob_btts_no:.1%} (Kurs: {1/max(prob_btts_no, 0.001):.2f})")

    st.divider()
    st.subheader("🎲 Symulacja Monte Carlo (1 000 000 scenariuszy)")
    if st.button(f"🚀 URUCHOM ANALIZĘ 1 000 000 SCENARIUSZY", use_container_width=True, key=f"sim_{league_name}"):
        with st.status("Trwa symulowanie (1 mln prób)...", expanded=True) as status:
            n_sim = 1000000
            sim_h = np.random.poisson(lambda_f, n_sim)
            sim_a = np.random.poisson(mu_f, n_sim)
            res_df = pd.DataFrame({'H': sim_h, 'A': sim_a, 'Total': sim_h + sim_a})
            most_common_row = res_df.groupby(['H', 'A']).size().idxmax()
            st.success(f"🏆 Najczęstszy wynik w symulacji: **{most_common_row[0]}:{most_common_row[1]}**")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sns.kdeplot(sim_h, fill=True, color="#1f77b4", label=h_team, bw_adjust=3, ax=ax2)
            sns.kdeplot(sim_a, fill=True, color="#ff7f0e", label=a_team, bw_adjust=3, ax=ax2)
            ax2.set_xlim(-0.5, 8.5) 
            ax2.set_title("Rozkład prawdopodobieństwa goli")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)
            status.update(label="Analiza zakończona!", state="complete")

    st.markdown("<br><hr><h2 style='text-align: center;'>💬 Ekspert AI: Analiza Wyników</h2>", unsafe_allow_html=True)
    if "HF_TOKEN" in st.secrets:
        client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
        current_context = f"MECZ: {h_team} vs {a_team}. Szanse: {p1:.1%} / {px:.1%} / {p2:.1%}. ExG: {lambda_f:.2f} - {mu_f:.2f}."
        if f"messages_{league_name}" not in st.session_state: st.session_state[f"messages_{league_name}"] = []
        for message in st.session_state[f"messages_{league_name}"]:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("Zadaj pytanie...", key=f"chat_input_{league_name}"):
            st.session_state[f"messages_{league_name}"].append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=[{"role": "system", "content": f"Ekspert piłkarski. Kontekst: {current_context}"}, {"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    full_response = response.choices[0].message.content
                    placeholder.markdown(full_response)
                    st.session_state[f"messages_{league_name}"].append({"role": "assistant", "content": full_response})
                except Exception as e: st.error(f"Błąd AI: {str(e)}")
    else: st.info("Dodaj HF_TOKEN do Secrets.")

    # --- MODUŁ: VALUE BET DETECTOR ---
    st.divider()
    st.markdown("<h2 style='text-align: center;'>⚖️ Porównanie z Rynkiem (Value Bet Detector)</h2>", unsafe_allow_html=True)
    
    odds_data = get_live_odds(LEAGUE_MAP.get(league_name))
    if odds_data:
        match = next((m for m in odds_data if h_team in m['home_team'] or a_team in m['away_team']), None)
        
        if match:
            try:
                bookie = match['bookmakers'][0]
                mkt = bookie['markets'][0]['outcomes']
                
                live_1 = next(o['price'] for o in mkt if o['name'] == match['home_team'])
                live_2 = next(o['price'] for o in mkt if o['name'] == match['away_team'])
                live_x = next(o['price'] for o in mkt if o['name'] == 'Draw')
                
                compare_df = pd.DataFrame({
                    "Typ": [f"1 ({h_team})", "X (Remis)", f"2 ({a_team})"],
                    "Twoje Prawd.": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
                    "Twój Kurs": [model_odds[0], model_odds[1], model_odds[2]],
                    "Kurs Bukmachera": [live_1, live_x, live_2],
                    "Value (%)": [(p1 * live_1 - 1), (px * live_x - 1), (p2 * live_2 - 1)]
                })

                def style_value(val):
                    color = '#28a745' if val > 0.05 else '#dc3545' if val < -0.05 else 'transparent'
                    return f'background-color: {color}; color: white; font-weight: bold'

                st.markdown(f"**Źródło danych:** {bookie['title']} (Aktualizacja: {match['commence_time'][:10]})")
                st.table(compare_df.style.applymap(style_value, subset=['Value (%)'])
                         .format("{:.2f}", subset=['Twój Kurs', 'Kurs Bukmachera'])
                         .format("{:.2%}", subset=['Value (%)']))
                
                if any((p1*live_1-1) > 0.05 for _ in [1]):
                     st.success("🎯 Sugestia: Znaleziono matematyczną przewagę nad bukmacherem!")
            except Exception:
                st.info("Nie udało się sparsować kursów dla tego meczu.")
        else:
            st.info("Brak aktywnych kursów w API dla wybranego zestawienia.")
    else:
        st.warning("Nie udało się pobrać kursów. Sprawdź ODDS_API_KEY w Secrets.")

# Wywołanie UI
with tab_bl: render_league_ui(load_bundesliga(), "Bundesliga")
with tab_pl: render_league_ui(load_premier_league(), "Premier League")
with tab_ll: render_league_ui(load_la_liga(), "La Liga")
