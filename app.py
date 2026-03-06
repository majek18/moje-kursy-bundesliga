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

# --- SCRAPER Z POPRAWKĄ TYPÓW DANYCH ---
@st.cache_data(ttl=3600)
def get_live_data():
    url = "https://understat.com/league/Bundesliga"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "lxml")
    scripts = soup.find_all('script')
    
    data_script = ""
    for s in scripts:
        if 'teamsData' in s.text:
            data_script = s.text
            break
            
    start_ind = data_script.find("('") + 2
    end_ind = data_script.find("')")
    json_data = data_script[start_ind:end_ind].encode('utf8').decode('unicode_escape')
    raw_data = json.loads(json_data)
    
    teams = []
    for id in raw_data:
        t_data = raw_data[id]
        team_name = t_data['title']
        history = t_data['history']
        
        # Konwersja danych na float (ważne!)
        for m in history:
            m['scored'] = float(m['scored'])
            m['missed'] = float(m['missed'])
            m['xG'] = float(m['xG'])
            m['xGA'] = float(m['xGA'])
        
        h_g = [m for m in history if m['h_a'] == 'h']
        a_g = [m for m in history if m['h_a'] == 'a']
        
        teams.append({
            'Team': team_name,
            'H_GF': np.mean([m['scored'] for m in h_g]) if h_g else 0,
            'H_GA': np.mean([m['missed'] for m in h_g]) if h_g else 0,
            'HxG_F': np.mean([m['xG'] for m in h_g]) if h_g else 0,
            'HxG_A': np.mean([m['xGA'] for m in h_g]) if h_g else 0,
            'A_GF': np.mean([m['scored'] for m in a_g]) if a_g else 0,
            'A_GA': np.mean([m['missed'] for m in a_g]) if a_g else 0,
            'AxG_F': np.mean([m['xG'] for m in a_g]) if a_g else 0,
            'AxG_A': np.mean([m['xGA'] for m in a_g]) if a_g else 0,
            'T_GF': np.mean([m['scored'] for m in history]),
            'T_GA': np.mean([m['missed'] for m in history]),
            'TxG_F': np.mean([m['xG'] for m in history]),
            'TxG_A': np.mean([m['xGA'] for m in history])
        })
    return pd.DataFrame(teams).sort_values('Team')

# Pobieranie danych
try:
    df = get_live_data()
    st.sidebar.success("🟢 Dane LIVE: Understat")
except Exception as e:
    st.sidebar.error(f"🔴 Błąd danych: {e}")
    st.stop()

# --- SIDEBAR WAGI ---
st.sidebar.header("⚖️ Konfiguracja Wag")
D_W = [40, 25, 20, 15]
options = list(range(0, 105, 5))

# Inicjalizacja session_state
keys = ['w_xg_dv', 'w_g_dv', 'w_xg_all', 'w_g_all']
for i, k in enumerate(keys):
    if k not in st.session_state:
        st.session_state[k] = D_W[i]

if st.sidebar.button("🔄 Resetuj wagi (40/25/20/15)"):
    for i, k in enumerate(keys): st.session_state[k] = D_W[i]
    st.rerun()

v0 = st.sidebar.selectbox("🎯 xG Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_xg_dv), key='w_xg_dv')
v1 = st.sidebar.selectbox("⚽ Gole Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w_g_dv), key='w_g_dv')
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(st.session_state.w_xg_all), key='w_xg_all')
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(st.session_state.w_g_all), key='w_g_all')

w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100
if (v0+v1+v2+v3) != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor Pro")
avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()

c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{logo_map.get(h_team, 0)}.png", width=120)
with c2:
    a_team = st.selectbox("Gość", df['Team'], index=min(1, len(df)-1))
    st.image(f"https://tmssl.akamaized.net/images/wappen/head/{logo_map.get(a_team, 0)}.png", width=120)

# --- OBLICZENIA ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

# Atak i Obrona
l_h = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3) / avg_h_gf
d_h = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3) / avg_a_gf
l_a = (a['AxG_F']*w0 + a['A_GF']*w1 + a['TxG_F']*w2 + a['T_GF']*w3) / avg_a_gf
d_a = (a['AxG_A']*w0 + a['A_GA']*w1 + a['TxG_A']*w2 + a['T_GA']*w3) / avg_h_gf

lambda_final = l_h * d_a * avg_h_gf
mu_final = l_a * d_h * avg_a_gf

matrix = np.outer(poisson.pmf(range(10), lambda_final), poisson.pmf(range(10), mu_final))
p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- WYNIKI ---
st.divider()
st.subheader("🎯 Prognoza Wyniku Końcowego")
m1, mx, m2 = st.columns(3)
m1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
mx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
m2.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

st.write("### 🏦 Kalkulator Value Bet")
ci1, ci2, ci3 = st.columns(3)
with ci1: bk1 = st.text_input(f"Kurs {h_team}", value="2.00")
with ci2: bkx = st.text_input("Kurs X", value="3.50")
with ci3: bk2 = st.text_input(f"Kurs {a_team}", value="4.00")

def get_v(prob, bk):
    try:
        k = float(bk.replace(',', '.'))
        return f"✅ TAK ({k:.2f})" if k > (1/prob) else f"❌ NIE ({k:.2f})"
    except: return "-"

st.table(pd.DataFrame({
    "Typ": ["1", "X", "2"],
    "Model (%)": [f"{p1:.1%}", f"{px:.1%}", f"{p2:.1%}"],
    "Kurs Fair": [f"{1/p1:.2f}", f"{1/px:.2f}", f"{1/p2:.2f}"],
    "Value?": [get_v(p1, bk1), get_v(px, bkx), get_v(p2, bk2)]
}))

with st.expander("⚽ Macierz Prawdopodobieństwa (0-6 goli)"):
    
    limit = 7
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix[:limit, :limit], annot=True, fmt=".1%", cmap="YlGn", cbar=False, linewidths=1.5)
    plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
    st.pyplot(fig)
