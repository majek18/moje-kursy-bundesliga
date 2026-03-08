import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
import requests

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor Pro", layout="wide", page_icon="⚽")

# --- MAPOWANIE LIG DLA API ---
LEAGUE_MAP = {
    "Bundesliga": "soccer_germany_bundesliga",
    "Premier League": "soccer_upcoming",
    "La Liga": "soccer_spain_la_liga"
}

# --- DANE Z OSTATNICH MECZÓW (NA PODSTAWIE SCREENÓW) ---
# Uwaga: Dane wyciągnięte bezpośrednio z Twoich tabel "Form" i "xG/xGA"
@st.cache_data
def load_recent_form_data():
    return {
        'Bayern Munich': {'GF': 20/6, 'GA': 8/6, 'xG': 16.10/5, 'xGA': 8.36/5, 'matches': 6},
        'Borussia Dortmund': {'GF': 15/6, 'GA': 9/6, 'xG': 8.22/5, 'xGA': 10.06/5, 'matches': 6},
        'Augsburg': {'GF': 9/6, 'GA': 7/6, 'xG': 10.02/5, 'xGA': 8.57/5, 'matches': 6},
        'VfB Stuttgart': {'GF': 14/6, 'GA': 8/6, 'xG': 9.62/5, 'xGA': 9.40/5, 'matches': 6},
        'RB Leipzig': {'GF': 11/6, 'GA': 9/6, 'xG': 13.56/5, 'xGA': 7.65/5, 'matches': 6},
        'Bayer Leverkusen': {'GF': 10/6, 'GA': 6/6, 'xG': 11.02/5, 'xGA': 7.04/5, 'matches': 6},
        'St. Pauli': {'GF': 7/6, 'GA': 9/6, 'xG': 4.00/5, 'xGA': 7.81/5, 'matches': 6},
        'Hamburger SV': {'GF': 9/6, 'GA': 7/6, 'xG': 8.49/5, 'xGA': 12.42/5, 'matches': 6},
        'Hoffenheim': {'GF': 13/6, 'GA': 11/6, 'xG': 11.76/5, 'xGA': 10.16/5, 'matches': 6},
        'Mainz 05': {'GF': 8/6, 'GA': 9/6, 'xG': 10.93/5, 'xGA': 6.86/5, 'matches': 6},
        'Eintracht Frankfurt': {'GF': 10/6, 'GA': 10/6, 'xG': 6.40/4, 'xGA': 4.98/4, 'matches': 6},
        'Freiburg': {'GF': 6/6, 'GA': 10/6, 'xG': 5.63/5, 'xGA': 7.91/5, 'matches': 6},
        'Borussia M.Gladbach': {'GF': 5/6, 'GA': 11/6, 'xG': 6.62/5, 'xGA': 7.28/5, 'matches': 6},
        'FC Koln': {'GF': 6/6, 'GA': 11/6, 'xG': 4.83/5, 'xGA': 10.96/5, 'matches': 6},
        'Werder Bremen': {'GF': 4/6, 'GA': 9/6, 'xG': 9.23/5, 'xGA': 5.34/5, 'matches': 6},
        'Union Berlin': {'GF': 5/6, 'GA': 11/6, 'xG': 4.21/5, 'xGA': 8.21/5, 'matches': 6},
        'FC Heidenheim': {'GF': 7/6, 'GA': 15/6, 'xG': 6.64/5, 'xGA': 11.98/5, 'matches': 6},
        'Wolfsburg': {'GF': 6/6, 'GA': 14/6, 'xG': 8.22/5, 'xGA': 10.54/5, 'matches': 6},
    }

# --- FUNKCJA OBLICZANIA BONUSU (MODUŁ ZADANY) ---
def calculate_form_bonus(team_name, base_stats, is_home):
    recent_db = load_recent_form_data()
    if team_name not in recent_db:
        return 0.0, 0.0, {} # Domyślnie brak bonusu dla PL/LL jeśli brak danych

    r = recent_db[team_name]
    # Wybór bazy w zależności od tego czy grają u siebie czy na wyjeździe
    base_xg_f = base_stats['HxG_F'] if is_home else base_stats['AxG_F']
    base_gf = base_stats['H_GF'] if is_home else base_stats['A_GF']
    base_xga = base_stats['HxG_A'] if is_home else base_stats['AxG_A']
    base_ga = base_stats['H_GA'] if is_home else base_stats['A_GA']

    # 1. Atak
    trend_atk = (r['xG'] - base_xg_f) / base_xg_f
    skut_atk = (r['GF'] - r['xG']) / r['xG']
    raw_atk_bonus = (trend_atk * 0.7) + (skut_atk * 0.3)
    final_atk_bonus = raw_atk_bonus * 0.25

    # 2. Obrona (tu spadek xGA/GA jest pozytywny, więc odwracamy logikę)
    trend_def = (base_xga - r['xGA']) / base_xga
    skut_def = (r['xGA'] - r['GA']) / r['xGA']
    raw_def_bonus = (trend_def * 0.7) + (skut_def * 0.3)
    final_def_bonus = raw_def_bonus * 0.25

    details = {
        'base_xg_f': base_xg_f, 'recent_xg_f': r['xG'], 'trend_atk': trend_atk,
        'recent_gf': r['GF'], 'skut_atk': skut_atk, 'final_atk': final_atk_bonus,
        'base_xga': base_xga, 'recent_xga': r['xGA'], 'trend_def': trend_def,
        'recent_ga': r['GA'], 'skut_def': skut_def, 'final_def': final_def_bonus,
        'matches': r['matches']
    }
    return final_atk_bonus, final_def_bonus, details

# --- POZOSTAŁE FUNKCJE BAZOWE (BEZ ZMIAN) ---
def get_live_odds(league_key):
    try:
        api_key = st.secrets["ODDS_API_KEY"]
        url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
        params = {"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else None
    except: return None

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
        'TxG_F': [3.07, 1.85, 1.85, 1.96, 2.20, 2.02, 1.56, 1.42, 1.32, 1.42, 1.32, 1.43, 1.45, 1.63, 0.97, 1.32, 1.41, 1.36],
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.06, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38],
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

# (Funkcje load_premier_league i load_la_liga bez zmian - pominięte dla zwięzłości w tym bloku)
# [TU WSTAW ORYGINALNE FUNKCJE load_premier_league I load_la_liga]

def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SESSION STATE & SIDEBAR ---
if 'mod_reset' not in st.session_state: st.session_state.mod_reset = 0
def reset_mods(): st.session_state.mod_reset += 1

st.sidebar.header("⚙️ Konfiguracja Wag")
if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0
def reset_weights(): st.session_state.reset_counter += 1
st.sidebar.button("🔄 Resetuj wagi", on_click=reset_weights)

options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", options, index=options.index(40), key=f"w0_{st.session_state.reset_counter}")
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", options, index=options.index(25), key=f"w1_{st.session_state.reset_counter}")
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", options, index=options.index(20), key=f"w2_{st.session_state.reset_counter}")
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, index=options.index(15), key=f"w3_{st.session_state.reset_counter}")

if v0 + v1 + v2 + v3 != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100
fixed_rho = -0.15

# --- INTERFEJS GŁÓWNY ---
tab_bl, tab_pl, tab_ll = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "🇪🇸 La Liga"])

def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    st.title(f"⚽ {league_name} Predictor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
        h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=80)
    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=1, key=f"a_{league_name}")
        a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=80)

    # Obliczenia bazowe
    h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
    l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['AxG_F']*w2 + h['T_GF']*w3)
    m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['AxG_A']*w2 + h['T_GA']*w3)
    l_a_r = (a['TxG_F']*w0 + a['A_GF']*w1 + a['AxG_F']*w2 + a['T_GF']*w3)
    m_a_r = (a['TxG_A']*w0 + a['A_GA']*w1 + a['AxG_A']*w2 + a['T_GA']*w3)

    # NOWY MODUŁ: BONUS FORMY (ZGODNIE ZE SCREENEM)
    st.divider()
    st.subheader("🚀 Bonus Formy (Trend & Skuteczność)")
    
    # Obliczamy dla obu
    h_atk_b, h_def_b, h_det = calculate_form_bonus(h_team, h, True)
    a_atk_b, a_def_b, a_det = calculate_form_bonus(a_team, a, False)

    # Wizualizacja dla obu zespołów w kolumnach
    b_col1, b_col2 = st.columns(2)
    
    for team, bonus, det, col, side in [(h_team, h_atk_b, h_det, b_col1, "Gospodarz"), (a_team, a_atk_b, a_det, b_col2, "Gość")]:
        with col:
            st.markdown(f"### Obliczenia dla {team} ({side})")
            if not det:
                st.info("Brak szczegółowych danych formy dla tego zespołu.")
                continue
                
            # Tabela wejściowa jak na screenie
            input_df = pd.DataFrame({
                "Parametr (na mecz)": ["Gole Strzelone (GF)", "Gole Stracone (GA)", "xG (Kreacja)", "xGA (Dopuszczone)"],
                "Sezon (Baza)": [f"{det['recent_gf']/ (1+det['skut_atk']):.2f}", f"{det['recent_ga']}", f"{det['base_xg_f']:.2f}", f"{det['base_xga']:.2f}"],
                f"Ostatnie {det['matches']} meczów": [f"{det['recent_gf']:.2f}", f"{det['recent_ga']:.2f}", f"{det['recent_xg_f']:.2f}", f"{det['recent_xga']:.2f}"]
            })
            st.table(input_df)

            st.markdown("**2. Obliczanie Bonusu Ataku**")
            st.write(f"* Trend Kreacji (70%): `({det['recent_xg_f']:.2f} - {det['base_xg_f']:.2f}) / {det['base_xg_f']:.2f} = {det['trend_atk']:.2%}`")
            st.write(f"* Skuteczność (30%): `({det['recent_gf']:.2f} - {det['recent_xg_f']:.2f}) / {det['recent_xg_f']:.2f} = {det['skut_atk']:.2%}`")
            st.write(f"* Surowy Bonus Ataku: `({det['trend_atk']:.2f} * 0.7) + ({det['skut_atk']:.2f} * 0.3) = {det['trend_atk']*0.7 + det['skut_atk']*0.3:.2%}`")
            st.success(f"Wynik: Siła ataku {team} zostaje zmieniona o **{det['final_atk']:.2%}**")

            st.markdown("**3. Obliczanie Bonusu Obrony**")
            st.write(f"* Trend Defensywny (70%): `({det['base_xga']:.2f} - {det['recent_xga']:.2f}) / {det['base_xga']:.2f} = {det['trend_def']:.2%}`")
            st.write(f"* Skuteczność GK (30%): `({det['recent_xga']:.2f} - {det['recent_ga']:.2f}) / {det['recent_xga']:.2f} = {det['skut_def']:.2%}`")
            st.success(f"Wynik: Obrona {team} jest oceniana o **{det['final_def']:.2%}** lepiej/gorzej")

    # Aplikacja bonusów do lambdy
    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)

    lambda_f = (h_atk_s * a_def_s * avg_h_gf) * (1 + h_atk_b) * (1 - a_def_b)
    mu_f = (a_atk_s * h_def_s * avg_a_gf) * (1 + a_atk_b) * (1 - h_def_b)

    # --- POISSON & RESZTA (BEZ ZMIAN W LOGICE) ---
    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, fixed_rho)
    matrix /= matrix.sum()

    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    model_odds = [1/max(p1, 0.001), 1/max(px, 0.001), 1/max(p2, 0.001)]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {model_odds[0]:.2f}")
    c2.metric("Remis", f"{px:.1%}", f"Kurs: {model_odds[1]:.2f}")
    c3.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {model_odds[2]:.2f}")

    # [RESZTA TWOJEGO KODU - WYKRESY, SYMULACJE, AI - BEZ ZMIAN]
    # ...
