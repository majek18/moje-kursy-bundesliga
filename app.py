import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Bundesliga Predictor", layout="wide")

st.title("⚽ Bundesliga Poisson & Dixon-Coles Predictor")
st.write("Model oblicza kursy na podstawie Goli i xG (Sezon 25/26)")

# --- DANE BUNDESLIGI ---
# Tu docelowo możesz aktualizować liczby
data = pd.DataFrame({
    'Team': ['Bayern Munich', 'Bayer Leverkusen', 'RB Leipzig', 'Dortmund', 'Stuttgart', 'Eintracht Frankfurt'],
    'GF_H': [2.8, 2.5, 2.1, 2.0, 1.9, 1.8],
    'xG_H': [2.7, 2.4, 2.0, 1.8, 2.1, 1.7],
    'GA_H': [0.7, 0.9, 1.0, 1.2, 1.1, 1.3],
    'xGA_H': [0.8, 1.0, 1.1, 1.3, 1.0, 1.4],
    'GF_A': [2.4, 2.1, 1.8, 1.5, 1.6, 1.4],
    'xG_A': [2.3, 2.0, 1.7, 1.4, 1.8, 1.5],
    'GA_A': [1.1, 1.2, 1.3, 1.6, 1.5, 1.7],
    'xGA_A': [1.0, 1.3, 1.4, 1.5, 1.3, 1.6]
})

# Panel boczny
st.sidebar.header("Ustawienia Modelu")
waga_xg = st.sidebar.slider("Waga xG", 0.0, 1.0, 0.6)
rho = -0.15 # Korekta Dixona-Colesa

# Wybór meczu
col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Gospodarz", data['Team'])
with col2:
    a_team = st.selectbox("Gość", data['Team'])

# Obliczenia
def get_stats(team):
    row = data[data['Team'] == team].iloc[0]
    # Miks goli i xG
    att_h = row['GF_H'] * (1-waga_xg) + row['xG_H'] * waga_xg
    def_h = row['GA_H'] * (1-waga_xg) + row['xGA_H'] * waga_xg
    att_a = row['GF_A'] * (1-waga_xg) + row['xG_A'] * waga_xg
    def_a = row['GA_A'] * (1-waga_xg) + row['xGA_A'] * waga_xg
    return att_h, def_h, att_a, def_a

att_h, def_h, _, _ = get_stats(h_team)
_, _, att_a, def_a = get_stats(a_team)

# Średnie ligowe (uproszczone)
avg_h, avg_a = 1.6, 1.3

lambda_h = (att_h / avg_h) * (def_a / avg_h) * avg_h
mu_a = (att_a / avg_a) * (def_h / avg_a) * avg_a

# Macierz i Dixon-Coles
max_g = 6
matrix = np.outer(poisson.pmf(range(max_g), lambda_h), poisson.pmf(range(max_g), mu_a))
matrix[0,0] *= (1 - lambda_h * mu_a * rho); matrix[1,0] *= (1 + lambda_h * rho)
matrix[0,1] *= (1 + mu_a * rho); matrix[1,1] *= (1 - rho)

# Wyniki
p1 = np.sum(np.tril(matrix, -1))
px = np.sum(np.diag(matrix))
p2 = np.sum(np.triu(matrix, 1))

st.divider()
res1, resx, res2 = st.columns(3)
res1.metric("Wygrana Gospodarza", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
resx.metric("Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
res2.metric("Wygrana Gościa", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")
