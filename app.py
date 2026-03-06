import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor Pro", layout="wide")

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
        'TxG_A': [1.13, 1.32, 1.59, 1.40, 1.42, 1.27, 1.61, 1.52, 1.88, 1.46, 1.72, 1.63, 1.89, 1.90, 1.83, 1.72, 1.96, 2.22],
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
        'A_GA': [0.81, 1.14, 1.60, 1.35, 1.20, 1.53, 1.66, 1.00, 2.21, 1.78, 1.40, 1.35, 1.13, 1.42, 2.00, 1.50, 1.60, 1.80, 2.33, 1.50],
        'AxG_F': [1.87, 1.78, 1.70, 1.32, 2.10, 1.81, 1.48, 1.22, 1.79, 1.11, 0.91, 1.03, 1.43, 1.48, 1.23, 1.10, 0.90, 1.20, 0.85, 0.68],
        'AxG_A': [0.84, 1.31, 1.51, 1.78, 1.41, 1.47, 1.62, 1.59, 2.20, 1.83, 1.75, 1.27, 1.49, 1.64, 1.77, 1.53, 1.85, 2.04, 2.43, 1.75],
        'Logo_ID': [11, 281, 985, 405, 631, 31, 1148, 29, 1003, 931, 289, 762, 873, 1237, 399, 148, 703, 379, 1132, 543]
    }
    return pd.DataFrame(data)

# --- FUNKCJA KOREKTY DIXON-COLES ---
def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# --- SIDEBAR: KONFIGURACJA WAG ---
st.sidebar.header("⚙️ Konfiguracja Wag")
if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0
def reset_weights(): st.session_state.reset_counter += 1
st.sidebar.button("🔄 Resetuj wagi", on_click=reset_weights)

options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", options, index=options.index(40), key=f"w0_{st.session_state.reset_counter}")
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", options, index=options.index(25), key=f"w1_{st.session_state.reset_counter}")
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", options, index=options.index(20), key=f"w2_{st.session_state.reset_counter}")
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, index=options.index(15), key=f"w3_{st.session_state.reset_counter}")

total_pct = v0 + v1 + v2 + v3
if total_pct != 100:
    st.sidebar.error(f"Suma: {total_pct}% (musi być 100%)")
    st.stop()

w0, w1, w2, w3 = v0/100, v1/100, v2/100, v3/100
fixed_rho = -0.15

# --- LOGIKA UI LIGI ---
def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    
    st.markdown(f"<h1 style='text-align: center;'>⚽ {league_name} Predictor</h1>", unsafe_allow_html=True)
    
    # Wybór drużyn i herby
    col_a, col_spacer, col_b = st.columns([4, 1, 4])
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
        h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
        st.markdown(f"<div style='text-align: center'><img src='https://tmssl.akamaized.net/images/wappen/head/{h_id}.png' width='110'></div>", unsafe_allow_html=True)
    
    with col_spacer:
        st.markdown("<h2 style='text-align: center; padding-top: 50px;'>VS</h2>", unsafe_allow_html=True)

    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=1, key=f"a_{league_name}")
        a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
        st.markdown(f"<div style='text-align: center'><img src='https://tmssl.akamaized.net/images/wappen/head/{a_id}.png' width='110'></div>", unsafe_allow_html=True)

    # Obliczenia parametrów
    h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
    l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3)
    m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3)
    l_a_r = (a['AxG_F']*w0 + a['A_GF']*w1 + a['TxG_F']*w2 + a['T_GF']*w3)
    m_a_r = (a['AxG_A']*w0 + a['A_GA']*w1 + a['TxG_A']*w2 + a['T_GA']*w3)

    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)
    lambda_f = h_atk_s * a_def_s * avg_h_gf
    mu_f = a_atk_s * h_def_s * avg_a_gf

    # Macierz Poissona
    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, fixed_rho)
    matrix /= matrix.sum()
    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

    # 1. SZANSE PROCENTOWE I KURSY
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.container(border=True).metric(f"🏠 {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
    with c2:
        st.container(border=True).metric("🤝 Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
    with c3:
        st.container(border=True).metric(f"🚀 {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

    # 2. TABELA PORÓWNANIA SIŁY
    st.markdown("### 📊 Porównanie Potencjału")
    def get_badge(val, reverse=False):
        if (not reverse and val >= 1.1) or (reverse and val <= 0.9): return "🟢 **MOCNY**"
        if (not reverse and val <= 0.9) or (reverse and val >= 1.1): return "🔴 **SŁABY**"
        return "🟡 **ŚREDNI**"

    st.markdown(f"""
    | Statystyka | {h_team} | {a_team} |
    | :--- | :---: | :---: |
    | **Atak (Skuteczność)** | {h_atk_s:.2f} ({get_badge(h_atk_s)}) | {a_atk_s:.2f} ({get_badge(a_atk_s)}) |
    | **Obrona (Stabilność)** | {h_def_s:.2f} ({get_badge(h_def_s, True)}) | {a_def_s:.2f} ({get_badge(a_def_s, True)}) |
    """)

    # 3. ŚCIEŻKA OBLICZENIOWA
    with st.expander("🧮 Szczegółowa Ścieżka Matematyczna"):
        st.write(f"Średnie ligowe: Gospodarze `{avg_h_gf:.3f}` | Goście `{avg_a_gf:.3f}`")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**{h_team}**")
            st.write(f"Atak: `{l_h_r:.3f}/{avg_h_gf:.3f} = {h_atk_s:.3f}`")
            st.write(f"Obrona: `{m_h_r:.3f}/{avg_a_gf:.3f} = {h_def_s:.3f}`")
        with sc2:
            st.markdown(f"**{a_team}**")
            st.write(f"Atak: `{l_a_r:.3f}/{avg_a_gf:.3f} = {a_atk_s:.3f}`")
            st.write(f"Obrona: `{m_a_r:.3f}/{avg_h_gf:.3f} = {a_def_s:.3f}`")
        st.latex(rf"\lambda = {lambda_f:.3f}, \quad \mu = {mu_f:.3f}")

    # 4. MACIERZ PRAWDOPODOBIEŃSTWA
    with st.expander("📊 Macierz Wyników (Dokładne Prawdopodobieństwo)"):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(matrix[:7, :7], annot=True, fmt=".1%", cmap="YlGn", cbar=False)
        plt.xlabel(f"Gole {a_team}")
        plt.ylabel(f"Gole {h_team}")
        st.pyplot(fig)

    # 5. RYNEK GOLI I BTTS
    st.markdown("### 🎯 Przewidywania Rynkowe")
    m1, m2 = st.columns(2)
    with m1:
        with st.container(border=True):
            st.write("**Linia 2.5 Gola**")
            p_over25 = 1 - sum(matrix[x, y] for x in range(max_g) for y in range(max_g) if x + y < 2.5)
            st.write(f"🟢 **OVER**: {p_over25:.1%} (k: {1/p_over25:.2f})")
            st.write(f"🔴 **UNDER**: {1-p_over25:.1%} (k: {1/(1-p_over25):.2f})")
    with m2:
        with st.container(border=True):
            st.write("**BTTS (Obie strzelą)**")
            p_btts = sum(matrix[x, y] for x in range(1, max_g) for y in range(1, max_g))
            st.write(f"✅ **TAK**: {p_btts:.1%} (k: {1/p_btts:.2f})")
            st.write(f"❌ **NIE**: {1-p_btts:.1%} (k: {1/(1-p_btts):.2f})")

    # 6. SYMULACJA MONTE CARLO
    st.divider()
    if st.button(f"🎲 SYMULUJ MECZ (10 000 prób)", use_container_width=True, key=f"sim_btn_{league_name}"):
        with st.status("Trwa symulowanie spotkania...", expanded=True) as status:
            n_sim = 10000
            sim_h = np.random.poisson(lambda_f, n_sim)
            sim_a = np.random.poisson(mu_f, n_sim)
            
            res = pd.DataFrame({'H': sim_h, 'A': sim_a})
            res['Score'] = res['H'].astype(str) + ":" + res['A'].astype(str)
            most_common = res['Score'].value_counts().idxmax()
            
            st.success(f"🏆 Najczęstszy wynik w symulacji: **{most_common}**")
            
            fig2, ax2 = plt.subplots(figsize=(10, 3.5))
            sns.kdeplot(sim_h, fill=True, color="#1f77b4", label=h_team, bw_adjust=2)
            sns.kdeplot(sim_a, fill=True, color="#ff7f0e", label=a_team, bw_adjust=2)
            plt.title("Gęstość prawdopodobieństwa goli")
            plt.xlabel("Gole")
            plt.legend()
            st.pyplot(fig2)
            status.update(label="Symulacja zakończona!", state="complete")

# --- URUCHOMIENIE APLIKACJI ---
tab_bl, tab_pl = st.tabs(["🇩🇪 Bundesliga", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League"])

with tab_bl:
    render_league_ui(load_bundesliga(), "Bundesliga")

with tab_pl:
    render_league_ui(load_premier_league(), "Premier League")
