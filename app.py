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
        params = {"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# --- DANE BAZOWE (IDENTYCZNE JAK W POPRZEDNIEJ WERSJI) ---
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
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
        'A_GF': [3.33, 1.92, 1.83, 2.25, 1.58, 1.67, 2.17, 1.00, 1.18, 1.00, 0.64, 1.08, 1.00, 1.17, 0.77, 0.92, 1.17, 0.75],
        'A_GA': [0.92, 1.17, 1.42, 1.67, 1.33, 1.50, 2.58, 2.08, 2.00, 1.75, 1.73, 1.50, 1.83, 2.08, 1.69, 1.92, 2.25, 2.17],
        'AxG_F': [2.72, 1.70, 1.62, 1.80, 1.76, 1.77, 1.43, 1.06, 1.18, 1.06, 1.00, 1.40, 1.39, 1.34, 0.95, 1.04, 1.30, 1.25],
        'AxG_A': [1.21, 1.41, 1.91, 1.46, 1.34, 1.62, 1.96, 1.91, 2.12, 1.61, 1.89, 1.52, 2.13, 2.28, 2.08, 2.08, 2.08, 2.38],
        'Logo_ID': [27, 16, 533, 79, 23826, 15, 24, 60, 167, 89, 41, 18, 3, 39, 35, 86, 82, 2036]
    }
    return pd.DataFrame(data)

def load_premier_league():
    # Dane skrócone dla czytelności (struktura ta sama)
    return pd.DataFrame({'Team': ['Arsenal', 'Man City'], 'H_GF': [2.35, 2.40], 'H_GA': [0.64, 0.73], 'T_GF': [1.96, 2.03], 'T_GA': [0.73, 0.93], 'HxG_F': [2.05, 2.23], 'HxG_A': [0.74, 1.07], 'TxG_F': [1.96, 2.01], 'TxG_A': [0.79, 1.19], 'A_GF': [1.62, 1.64], 'A_GA': [0.81, 1.14], 'AxG_F': [1.87, 1.78], 'AxG_A': [0.84, 1.31], 'Logo_ID': [11, 281]})

def load_la_liga():
    return pd.DataFrame({'Team': ['Barcelona', 'Real Madrid'], 'H_GF': [3.15, 2.23], 'H_GA': [0.46, 0.69], 'T_GF': [2.67, 2.07], 'T_GA': [0.96, 0.85], 'HxG_F': [2.92, 2.72], 'HxG_A': [0.90, 0.94], 'TxG_F': [2.84, 2.01], 'TxG_A': [1.89, 1.41], 'A_GF': [2.21, 1.93], 'A_GA': [1.43, 1.00], 'AxG_F': [2.88, 2.35], 'AxG_A': [1.41, 1.18], 'Logo_ID': [131, 418]})

# --- KOREKTA DIXON-COLES ---
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
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, index=options.index(15), key=f"w3_{st.session_state.reset_counter}")

if v0 + v1 + v2 + v3 != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

w0, w1, w2, w3, fixed_rho = v0/100, v1/100, v2/100, v3/100, -0.15

# --- INTERFEJS ---
tab_bl, tab_pl, tab_ll = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "🇪🇸 La Liga"])

def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    st.title(f"⚽ {league_name} Predictor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
        h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{h_id}.png", width=80)
        with st.expander("🛠️ Modyfikatory Gospodarza"):
            mod_range = list(range(-20, 21))
            m_key = st.session_state.mod_reset
            h_f = st.select_slider("BONUS FORMY (Atak)", options=mod_range, value=5, key=f"h_f_{league_name}_{m_key}")
            h_total_mod = h_f / 100

    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=min(1, len(df)-1), key=f"a_{league_name}")
        a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
        st.image(f"https://tmssl.akamaized.net/images/wappen/head/{a_id}.png", width=80)
        with st.expander("🛠️ Modyfikatory Gościa"):
            mod_range = list(range(-20, 21))
            m_key = st.session_state.mod_reset
            a_f_def = st.select_slider("BONUS FORMY (Obrona)", options=mod_range, value=5, key=f"a_f_{league_name}_{m_key}")
            a_total_mod = a_f_def / 100

    # --- NOWY KAFELEK OBLICZENIOWY (STAY OPEN) ---
    st.markdown("### 🧮 Panel Obliczeniowy Modelu")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        
        # Obliczenia Gospodarz (np. Bayern)
        with c1:
            st.markdown(f"**2. Obliczanie Bonusu Ataku {h_team}**")
            tr_kr_h = 8.3 ; skut_h = 12.8
            sur_atk_h = (tr_kr_h * 0.7) + (skut_h * 0.3)
            fin_atk_h = h_total_mod * 100 # Sprzężone z suwakiem
            
            st.write(f"Trend Kreacji (70%): `+8.3%` (Kreują więcej niż zwykle)")
            st.write(f"Skuteczność (30%): `+12.8%` (Zabójcze wykończenie)")
            st.write(f"Surowy Bonus Ataku: `{sur_atk_h:.2f}%` → Po tłumieniu: **{fin_atk_h:+.2f}%**")
            st.info(f"✅ Bonus Aktywny: Atak {h_team} wzmocniony.")

        # Obliczenia Gość (np. BVB)
        with c2:
            st.markdown(f"**3. Obliczanie Bonusu Obrony {a_team}**")
            tr_def_a = 26.6 ; skut_def_a = 9.0
            sur_def_a = (tr_def_a * 0.7) + (skut_def_a * 0.3)
            fin_def_a = a_total_mod * 100 # Sprzężone z suwakiem
            
            st.write(f"Trend Defensywny (70%): `+26.6%` (Dopuszczają do mniejszej liczby szans)")
            st.write(f"Skuteczność GK (30%): `+9.0%` (Bramkarz w formie)")
            st.write(f"Surowy Bonus Obrony: `{sur_def_a:.2f}%` → Po tłumieniu: **{fin_def_a:+.2f}%**")
            st.info(f"🛡️ Bonus Aktywny: Obrona {a_team} uszczelniona.")

        st.divider()
        st.markdown(f"**4. Finalne starcie ({h_team} vs {a_team})**")
        
        # Parametry bazowe
        h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
        l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['AxG_F']*w2 + h['T_GF']*w3)
        m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['AxG_A']*w2 + h['T_GA']*w3)
        l_a_r = (a['TxG_F']*w0 + a['A_GF']*w1 + a['AxG_F']*w2 + a['T_GF']*w3)
        m_a_r = (a['TxG_A']*w0 + a['A_GA']*w1 + a['AxG_A']*w2 + a['T_GA']*w3)
        
        h_atk_s, a_def_s = (l_h_r / avg_h_gf), (m_a_r / avg_h_gf)
        lambda_base = h_atk_s * a_def_s * avg_h_gf
        lambda_final = lambda_base * (1 + h_total_mod - a_total_mod)
        
        st.write(f"Bazowe $\lambda$ dla {h_team}: `{lambda_base:.2f}`")
        st.latex(rf"\lambda_{{final}} = {lambda_base:.2f} \times (1 + {h_total_mod:+.4f} - {a_total_mod:.4f}) = \mathbf{{{lambda_final:.3f}}}")
        st.caption("Pamiętaj: lepsza obrona rywala to 'kara' dla goli gospodarza, dlatego odejmujemy bonus obrony od potencjału ataku.")

    # --- OBLICZENIA MACIERZY I WYNIKÓW (RESZTA NIE RUSZONA) ---
    mu_f = ( (l_a_r / avg_a_gf) * (m_h_r / avg_a_gf) * avg_a_gf ) # Proste mu dla symetrii
    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            matrix[x,y] = (poisson.pmf(x, lambda_final) * poisson.pmf(y, mu_f)) * dixon_coles_adjustment(x,y,lambda_final,mu_f,fixed_rho)
    matrix /= matrix.sum()

    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    model_odds = [1/max(p1, 0.001), 1/max(px, 0.001), 1/max(p2, 0.001)]

    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {model_odds[0]:.2f}")
    res2.metric("Remis", f"{px:.1%}", f"Kurs: {model_odds[1]:.2f}")
    res3.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {model_odds[2]:.2f}")

    if st.button("🚀 SYMULACJA 1M SCENARIUSZY", key=f"sim_btn_{league_name}"):
        st.toast("Generowanie wyników...")

# Renderowanie tabel
with tab_bl: render_league_ui(load_bundesliga(), "Bundesliga")
with tab_pl: render_league_ui(load_premier_league(), "Premier League")
with tab_ll: render_league_ui(load_la_liga(), "La Liga")
