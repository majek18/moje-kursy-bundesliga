import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundesliga Analyzer", layout="wide")

# --- ZAŁADOWANIE DANYCH ---
@st.cache_data
def load_data():
    # Dane na podstawie Twojego pliku (uśrednione/sezonowe)
    data = {
        'Team': ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 'VfB Stuttgart', 'Eintracht Frankfurt'],
        'H_GF': [4.00, 2.33, 2.08, 2.25, 1.75, 1.83], # Gole strzelone u siebie
        'H_GA': [1.00, 0.92, 0.92, 1.42, 1.00, 1.50], # Gole stracone u siebie
        'A_GF': [3.33, 1.92, 1.67, 1.58, 2.25, 2.17], # Gole strzelone na wyjeździe
        'A_GA': [0.92, 1.17, 1.50, 1.33, 1.67, 2.58], # Gole stracone na wyjeździe
        'Logo_ID': [27, 16, 15, 23826, 79, 24]
    }
    return pd.DataFrame(data)

df = load_data()

# --- ŚREDNIE LIGOWE (Kluczowe dla Twojego modelu) ---
AVG_H_GF = df['H_GF'].mean()
AVG_A_GF = df['A_GF'].mean()

# --- BOCZNY PANEL: TYLKO WAGI ---
st.sidebar.header("⚙️ Konfiguracja Wag")
if st.sidebar.button("🔄 Resetuj tylko wagi"):
    if 'weight' in st.session_state: del st.session_state['weight']
    st.rerun()

weight = st.sidebar.slider("Wpływ aktualnej formy vs historia", 0.0, 1.0, 0.5, key='weight')

# --- WYBÓR MECZU ---
st.title("⚽ Modelowanie Wyniku: Bundesliga")
col1, col2 = st.columns(2)

with col1:
    h_team = st.selectbox("Gospodarz", df['Team'], index=0)
    h_data = df[df['Team'] == h_team].iloc[0]
with col2:
    a_team = st.selectbox("Gość", df['Team'], index=1)
    a_data = df[df['Team'] == a_team].iloc[0]

# --- OBLICZENIA (TWOJA LOGIKA) ---
# 1. Siła ataku i obrony (Strength)
h_atk_strength = h_data['H_GF'] / AVG_H_GF
h_def_strength = h_data['H_GA'] / AVG_A_GF
a_atk_strength = a_data['A_GF'] / AVG_A_GF
a_def_strength = a_data['A_GA'] / AVG_H_GF

# 2. Wyliczenie Lambda (Gospodarz) i Mu (Gość)
lambda_h = h_atk_strength * a_def_strength * AVG_H_GF
mu_a = a_atk_strength * h_def_strength * AVG_A_GF

# --- TABELA WSPÓŁCZYNNIKÓW SIŁY ---
st.write("### 📊 Współczynniki Siły i Prognoza Goli")
strength_df = pd.DataFrame({
    "Kategoria": ["Atak (Offensive)", "Obrona (Defensive)", "Prognozowane Gole (xG)"],
    f"🏠 {h_team}": [f"{h_atk_strength:.2f}", f"{h_def_strength:.2f}", f"**{lambda_h:.2f}**"],
    f"✈️ {a_team}": [f"{a_atk_strength:.2f}", f"{a_def_strength:.2f}", f"**{mu_a:.2f}**"]
})
st.table(strength_df)

# --- MACIERZ POISSONA ---
matrix = np.outer(poisson.pmf(range(6), lambda_h), poisson.pmf(range(6), mu_a))
p_win = np.sum(np.tril(matrix, -1))
p_draw = np.sum(np.diag(matrix))
p_loss = np.sum(np.triu(matrix, 1))

# --- PORÓWNANIE KURSÓW ---
st.divider()
st.write("### 🏦 Porównywarka Kursów")

# Twoje kursy wynikające z prawdopodobieństwa
my_k1 = 1/p_win if p_win > 0 else 0
my_kx = 1/p_draw if p_draw > 0 else 0
my_k2 = 1/p_loss if p_loss > 0 else 0

bookmakers = {
    "Twoje Obliczenia": [my_k1, my_kx, my_k2],
    "STS": [1.85, 3.90, 4.20],
    "Fortuna": [1.82, 4.05, 4.35],
    "Superbet": [1.90, 3.80, 4.10]
}

comparison_rows = []
for name, odds in bookmakers.items():
    row = {
        "Bukmacher": name,
        "1 (Gosp)": round(odds[0], 2),
        "X (Remis)": round(odds[1], 2),
        "2 (Gość)": round(odds[2], 2),
        "Value?": "---" if name == "Twoje Obliczenia" else ("TAK" if any(odds[i] > [my_k1, my_kx, my_k2][i] for i in range(3)) else "NIE")
    }
    comparison_rows.append(row)

st.table(pd.DataFrame(comparison_rows))
st.info("💡 **Value: TAK** oznacza, że kurs u bukmachera jest wyższy niż Twoje matematyczne wyliczenie (warto grać).")

# Wizualizacja prawdopodobieństwa wyników
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(matrix, annot=True, fmt=".1%", cmap="Greens", xticklabels=range(6), yticklabels=range(6))
plt.title("Rozkład prawdopodobieństwa wyników (Gospodarz vs Gość)")
st.pyplot(fig)
