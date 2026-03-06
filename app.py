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

# --- FUNKCJA POBIERANIA DANYCH LIVE (Understat) ---
@st.cache_data(ttl=3600) # Dane odświeżają się co godzinę
def get_understat_data():
    url = "https://understat.com/league/Bundesliga"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "lxml")
    scripts = soup.find_all('script')
    
    # Szukamy danych tabeli w skryptach JS na stronie
    data_script = ""
    for s in scripts:
        if 'teamsData' in s.text:
            data_script = s.text
            break
            
    # Wyciąganie JSONa z tekstu skryptu
    start_ind = data_script.find("('") + 2
    end_ind = data_script.find("')")
    json_data = data_script[start_ind:end_ind]
    json_data = json_data.encode('utf8').decode('unicode_escape')
    raw_data = json.loads(json_data)
    
    # Przetwarzanie na czytelny DataFrame
    teams = []
    for id in raw_data:
        t_data = raw_data[id]
        team_name = t_data['title']
        history = t_data['history']
        
        # Obliczamy średnie z historii meczów
        h_games = [m for m in history if m['h_a'] == 'h']
        a_games = [m for m in history if m['h_a'] == 'a']
        
        teams.append({
            'Team': team_name,
            # Gole i xG Dom
            'H_GF': np.mean([m['scored'] for m in h_games]),
            'H_GA': np.mean([m['missed'] for m in h_games]),
            'HxG_F': np.mean([m['xG'] for m in h_games]),
            'HxG_A': np.mean([m['xGA'] for m in h_games]),
            # Gole i xG Wyjazd
            'A_GF': np.mean([m['scored'] for m in a_games]),
            'A_GA': np.mean([m['missed'] for m in a_games]),
            'AxG_F': np.mean([m['xG'] for m in a_games]),
            'AxG_A': np.mean([m['xGA'] for m in a_games]),
            # Ogólne (Sezon)
            'T_GF': np.mean([m['scored'] for m in history]),
            'TxG_F': np.mean([m['xG'] for m in history]),
            'T_GA': np.mean([m['missed'] for m in history]),
            'TxG_A': np.mean([m['xGA'] for m in history])
        })
    
    return pd.DataFrame(teams).sort_values('Team')

# Pobieranie danych
try:
    df = get_understat_data()
    st.sidebar.success("✅ Dane zaktualizowane z Understat.com")
except Exception as e:
    st.sidebar.error("⚠️ Błąd pobierania danych. Używam danych awaryjnych.")
    # Tutaj można wstawić Twój stary słownik z danymi jako backup

avg_h_gf = df['H_GF'].mean()
avg_a_gf = df['A_GF'].mean()

# --- SIDEBAR KONFIGURACJA ---
st.sidebar.header("⚖️ Konfiguracja Wag")
D_W = [40, 25, 20, 15]
options = [i for i in range(0, 105, 5)]

if 'w0' not in st.session_state:
    st.session_state.w0, st.session_state.w1, st.session_state.w2, st.session_state.w3 = D_W

if st.sidebar.button("🔄 Resetuj wagi"):
    st.session_state.w0, st.session_state.w1, st.session_state.w2, st.session_state.w3 = D_W
    st.rerun()

v0 = st.sidebar.selectbox("🎯 xG Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w0), key='w0')
v1 = st.sidebar.selectbox("⚽ Gole Sezon (dom/wyjazd) %", options, index=options.index(st.session_state.w1), key='w1')
v2 = st.sidebar.selectbox("📊 xG Sezon (cały) %", options, index=options.index(st.session_state.w2), key='w2')
v3 = st.sidebar.selectbox("📉 Gole Sezon (cały) %", options, index=options.index(st.session_state.w3), key='w3')

w_xg_dv, w_g_dv, w_xg_all, w_g_all = v0/100, v1/100, v2/100, v3/100
if (v0+v1+v2+v3) != 100:
    st.sidebar.error("Suma wag musi być 100%!")
    st.stop()

# --- WYBÓR MECZU ---
st.title("⚽ Bundesliga Predictor LIVE")
c1, c2 = st.columns(2)
with c1: h_team = st.selectbox("Gospodarz", df['Team'], index=0)
with c2: a_team = st.selectbox("Gość", df['Team'], index=1)

# --- OBLICZENIA ---
h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]

l_h = (h['HxG_F']*w_xg_dv + h['H_GF']*w_g_dv + h['TxG_F']*w_xg_all + h['T_GF']*w_g_all)
m_h = (h['HxG_A']*w_xg_dv + h['H_GA']*w_g_dv + h['TxG_A']*w_xg_all + h['T_GA']*w_g_all)
l_a = (a['AxG_F']*w_xg_dv + a['A_GF']*w_g_dv + a['TxG_F']*w_xg_all + a['T_GF']*w_g_all)
m_a = (a['AxG_A']*w_xg_dv + a['A_GA']*w_g_dv + a['TxG_A']*w_xg_all + a['T_GA']*w_g_all)

h_atk_s, h_def_s = (l_h / avg_h_gf), (m_h / avg_a_gf)
a_atk_s, a_def_s = (l_a / avg_a_gf), (m_a / avg_h_gf)

lambda_final = h_atk_s * a_def_s * avg_h_gf
mu_final = a_atk_s * h_def_s * avg_a_gf

matrix = np.outer(poisson.pmf(range(10), lambda_final), poisson.pmf(range(10), mu_final))
p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

# --- WIDOK ---
st.divider()
st.subheader("🎯 Prognoza Wyniku")
m1, mx, m2 = st.columns(3)
m1.metric(f"Wygrana {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
mx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
m2.metric(f"Wygrana {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

st.write("### 🏦 Kalkulator Value Bet")
ci1, ci2, ci3 = st.columns(3)
with ci1: bk1 = st.text_input(f"Kurs {h_team}", placeholder="1.85")
with ci2: bkx = st.text_input("Kurs X", placeholder="3.40")
with ci3: bk2 = st.text_input(f"Kurs {a_team}", placeholder="4.50")

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
    m_plot = matrix[:7, :7]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(m_plot, annot=True, fmt=".1%", cmap="YlGn", cbar=False, linewidths=1.5)
    plt.xlabel(f"Gole {a_team}"); plt.ylabel(f"Gole {h_team}")
    st.pyplot(fig)
