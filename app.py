import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json

st.set_page_config(page_title="Bundesliga Predictor LIVE", layout="wide")

# --- MAPOWANIE LOGOTYPÓW (Understat Name -> Transfermarkt ID) ---
logo_map = {
    'Bayern Munich': 27, 'Borussia Dortmund': 16, 'Bayer Leverkusen': 15,
    'RB Leipzig': 23826, 'Eintracht Frankfurt': 24, 'Stuttgart': 79,
    'Freiburg': 60, 'Wolfsburg': 82, 'Hoffenheim': 533, 'Mainz 05': 39,
    'Werder Bremen': 86, 'Union Berlin': 89, 'Augsburg': 167,
    'Borussia M.Gladbach': 18, 'Heidenheim': 2036, 'FC Cologne': 3,
    'St. Pauli': 35, 'Hamburger SV': 41, 'Holstein Kiel': 1297, 'Bochum': 80
}

# --- FUNKCJA POBIERANIA DANYCH Z UNDERSTAT ---
@st.cache_data(ttl=3600)
def get_live_data():
    url = "https://understat.com/league/Bundesliga"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content, "lxml")
    scripts = soup.find_all('script')
    
    data_script = ""
    for s in scripts:
        if 'teamsData' in s.text:
            data_script = s.text
            break
            
    if not data_script:
        raise ValueError("Nie znaleziono danych na stronie Understat.")

    # Wycinanie JSONa ze skryptu JS
    start_ind = data_script.find("('") + 2
    end_ind = data_script.find("')")
    json_data = data_script[start_ind:end_ind].encode('utf8').decode('unicode_escape')
    raw_data = json.loads(json_data)
    
    teams = []
    for tid in raw_data:
        t_info = raw_data[tid]
        team_name = t_info['title']
        history = t_info['history']
        
        # Konwersja na floaty, bo Understat trzyma to jako stringi
        for m in history:
            for key in ['scored', 'missed', 'xG', 'xGA']:
                m[key] = float(m[key])
        
        h_games = [m for m in history if m['h_a'] == 'h']
        a_games = [m for m in history if m['h_a'] == 'a']
        
        teams.append({
            'Team': team_name,
            'H_GF': np.mean([m['scored'] for m in h_games]) if h_games else 0,
            'H_GA': np.mean([m['missed'] for m in h_games]) if h_games else 0,
            'HxG_F': np.mean([m['xG'] for m in h_games]) if h_games else 0,
            'HxG_A': np.mean([m['xGA'] for m in h_games]) if h_games else 0,
            'A_GF': np.mean([m['scored'] for m in a_games]) if a_games else 0,
            'A_GA': np.mean([m['missed'] for m in a_games]) if a_games else 0,
            'AxG_F': np.mean([m['xG'] for m in a_games]) if a_games else 0,
            'AxG_A': np.mean([m['xGA'] for m in a_games]) if a_games else 0,
            'T_GF': np.mean([m['scored'] for m in history]),
            'T_GA': np.mean([m['missed'] for m in history]),
            'TxG_F': np.mean([m['xG'] for m in history]),
            'TxG_A': np.mean([m['xGA'] for m in history])
        })
    return pd.DataFrame(teams).sort_values('Team')

# --- INICJALIZACJA DANYCH ---
try:
    df = get_live_data()
    st.sidebar.success("🟢 Połączono z Understat (Dane LIVE)")
except Exception as e:
    st.sidebar.error(f"🔴 Błąd scrapowania: {e}")
    st.stop()

# --- SIDEBAR: KONFIGURACJA WAG ---
st.sidebar.header("⚖️ Konfiguracja Wag")
D_W = [40, 25, 20, 15] # Twoje domyślne wagi
options = list(range(0, 105, 5))

# Klucze do session_state
keys = ['w_xg_dv', 'w_g_dv', 'w_xg_all', 'w_g_all']
for i, k in enumerate(keys):
    if k not in st.session_state:
        st.session_state[k] = D_W[i]

if st.sidebar.button("🔄 Resetuj do: 40/25/20/15"):
    for i, k in enumerate(keys): st.session_state[k] = D_W[i]
    st.rerun()

v0 = st.sidebar.selectbox("🎯 xG Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_xg_dv), key='w_xg_dv')
v1 = st.sidebar.selectbox("⚽ Gole Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_g_dv), key='w_g_dv')
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(st.session_state.w_xg_all), key='w_xg_all')
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(st.session_state.w_g_all), key='w_g_all')

# Walidacja sumy
w_total = v0 + v1 + v2 + v3
if w_total != 100:
    st.sidebar.error(f"Suma wag wynosi {w_total}%. Musi być 100%!")
    st.stop()

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor LIVE")
avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{logo_map.get(h_team, 0)}.png", width=100)
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=min(1, len(df)-1))
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{logo_map.get(a_team, 0)}.png", width=100)

# --- OBLICZENIA SIŁY (STRENGTH) ---
h = df[df['Team'] == h_team].iloc[0]
a = df[df['Team'] == a_team].iloc[0]

# Wagi ułamkowe
w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100

# Obliczanie parametrów ataku i obrony na podstawie Twoich wag
h_atk = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3) / avg_h_gf
h_def = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3) / avg_a_gf
a_atk = (a['AxG_F']*w0 + a['A_GF']*w1 + a['TxG_F']*w2 + a['T_GF']*w3) / avg_a_gf
a_def = (a['AxG_A']*w0 + a['A_GA']*w1 + a['TxG_A']*w2 + a['T_GA']*w3) / avg_h_gf

# Średnia liczba goli (Lambda i Mu)
l_h = h_atk * a_def * avg_h_gf
l_a = a_atk * h_def * avg_a_gf

# Generowanie macierzy prawdopodobieństwa
max_g = 10
matrix = np.outer(poisson.pmf(range(max_g), l_h), poisson.pmf(range(max_g), l_a))
p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- SEKCA: DUŻE PROCENTY ---
st.divider()
st.subheader("🎯 Prognozowane Szanse")
m1, mx, m2 = st.columns(3)
m1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
mx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
m2.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

# --- TABELA VALUE BET ---
st.write("### 🏦 Kalkulator Value")
c_v1, c_v2, c_v3 = st.columns(3)
with c_v1: b1 = st.text_input(f"Kurs {h_team}", "2.10")
with c_v2: bx = st.text_input("Kurs X", "3.60")
with c_v3: b2 = st.text_input(f"Kurs {a_team}", "4.20")

def check_value(prob, book_odds):
    try:
        bo = float(book_odds.replace(',', '.'))
        fair = 1/prob
        return f"✅ TAK ({bo:.2f})" if bo > fair else f"❌ NIE ({bo:.2f})"
    except: return "-"

st.table(pd.DataFrame({
    "Typ": ["1", "X", "2"],
    "Model %": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
    "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"],
    "Value?": [check_value(p1, b1), check_value(px, bx), check_value(p2, b2)]
}))

# --- MACIERZ WYNIKÓW (ZIELONA) ---
with st.expander("⚽ Rozkład prawdopodobieństwa wyników (0-6 goli)"):
    
    limit = 7
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".1%", cmap="YlGn", cbar=False, linewidths=1)
    plt.xlabel(f"Gole {a_team}")
    plt.ylabel(f"Gole {h_team}")
    st.pyplot(fig)
