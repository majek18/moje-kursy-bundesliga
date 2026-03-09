import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor", layout="wide", page_icon="⚽")

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

# --- DANE RECENT BONUS: BUNDESLIGA ---
@st.cache_data
def load_bundesliga_recent_bonus_data():
    data = {
        'Team': [
            'Bayern Munich', 'Borussia Dortmund', 'Augsburg', 'VfB Stuttgart', 'RB Leipzig', 'Hoffenheim',
            'Hamburger SV', 'St. Pauli', 'Bayer Leverkusen', 'Mainz 05', 'Eintracht Frankfurt', 'Freiburg',
            'Borussia M.Gladbach', 'FC Cologne', 'Werder Bremen', 'Union Berlin', 'FC Heidenheim', 'Wolfsburg'
        ],
        'recent_form_matches': [6] * 18,
        'recent_GF_total': [20, 15, 9, 14, 11, 13, 9, 7, 10, 8, 10, 6, 5, 6, 4, 5, 7, 6],
        'recent_GA_total': [8, 9, 7, 8, 9, 11, 7, 9, 6, 9, 10, 10, 11, 11, 9, 11, 15, 14],
        'recent_xg_matches': [5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 4, 5, 5, 5, 5, 4, 5, 5],
        'recent_xG_total': [16.10, 8.22, 10.02, 9.62, 13.56, 11.76, 8.49, 4.00, 11.02, 10.93, 6.40, 5.63, 6.62, 4.83, 9.23, 4.21, 6.64, 8.22],
        'recent_xGA_total': [8.36, 10.06, 8.57, 9.40, 7.65, 10.16, 12.42, 7.81, 7.04, 6.86, 4.98, 7.91, 7.28, 10.96, 5.34, 8.21, 11.98, 10.54]
    }
    recent_df = pd.DataFrame(data)
    recent_df['recent_GF_pm'] = recent_df['recent_GF_total'] / recent_df['recent_form_matches']
    recent_df['recent_GA_pm'] = recent_df['recent_GA_total'] / recent_df['recent_form_matches']
    recent_df['recent_xG_pm'] = recent_df['recent_xG_total'] / recent_df['recent_xg_matches']
    recent_df['recent_xGA_pm'] = recent_df['recent_xGA_total'] / recent_df['recent_xg_matches']
    return recent_df

# --- DANE RECENT BONUS: PREMIER LEAGUE ---
@st.cache_data
def load_premier_league_recent_bonus_data():
    data = {
        'Team': [
            'Arsenal', 'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Brentford',
            'Bournemouth', 'Everton', 'Crystal Palace', 'Wolves', 'West Ham', 'Brighton',
            'Sunderland', 'Fulham', 'Newcastle', 'Leeds', 'Aston Villa', 'Burnley',
            'Nottingham Forest', 'Tottenham'
        ],
        'recent_form_matches': [6] * 20,
        'recent_GF_total': [13, 12, 10, 13, 14, 9, 6, 9, 9, 7, 8, 5, 6, 8, 10, 6, 4, 7, 5, 6],
        'recent_GA_total': [5, 6, 6, 7, 9, 8, 3, 7, 7, 7, 9, 5, 8, 11, 14, 10, 9, 14, 9, 15],
        'recent_xg_matches': [6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5],
        'recent_xG_total': [9.54, 11.51, 8.51, 11.02, 14.23, 8.27, 7.90, 8.35, 7.39, 5.52, 7.62, 4.71, 4.80, 8.37, 9.38, 4.83, 4.86, 5.14, 6.69, 4.91],
        'recent_xGA_total': [5.07, 5.44, 4.81, 6.03, 6.20, 8.62, 7.07, 6.95, 6.75, 14.98, 5.83, 4.54, 9.16, 7.77, 9.78, 7.78, 8.48, 9.37, 6.73, 12.19]
    }
    recent_df = pd.DataFrame(data)
    recent_df['recent_GF_pm'] = recent_df['recent_GF_total'] / recent_df['recent_form_matches']
    recent_df['recent_GA_pm'] = recent_df['recent_GA_total'] / recent_df['recent_form_matches']
    recent_df['recent_xG_pm'] = recent_df['recent_xG_total'] / recent_df['recent_xg_matches']
    recent_df['recent_xGA_pm'] = recent_df['recent_xGA_total'] / recent_df['recent_xg_matches']
    return recent_df

# --- DANE RECENT BONUS: LA LIGA ---
@st.cache_data
def load_la_liga_recent_bonus_data():
    data = {
        'Team': [
            'Barcelona', 'Getafe', 'Real Madrid', 'Villarreal', 'Athletic Club', 'Real Betis',
            'Atletico Madrid', 'Rayo Vallecano', 'Osasuna', 'Valencia', 'Celta Vigo',
            'Real Sociedad', 'Sevilla', 'Girona', 'Alaves', 'Mallorca', 'Levante',
            'Real Oviedo', 'Elche', 'Espanyol'
        ],
        'recent_form_matches': [6] * 20,
        'recent_GF_total': [15, 7, 11, 11, 10, 8, 8, 10, 8, 8, 8, 11, 7, 7, 7, 7, 5, 5, 6, 10],
        'recent_GA_total': [4, 2, 6, 9, 7, 7, 8, 5, 7, 8, 7, 12, 9, 8, 11, 11, 11, 12, 12, 17],
        'recent_xg_matches': [5, 5, 5, 6, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 6, 4, 5, 4],
        'recent_xG_total': [13.53, 4.95, 8.79, 10.96, 8.84, 7.76, 9.10, 8.18, 6.47, 7.06, 7.36, 7.45, 3.40, 11.04, 9.44, 3.55, 7.04, 5.12, 7.65, 3.78],
        'recent_xGA_total': [6.91, 7.54, 5.93, 10.22, 4.18, 4.14, 6.45, 5.47, 6.70, 6.75, 3.39, 10.23, 5.94, 7.31, 9.49, 11.49, 13.38, 7.38, 9.11, 9.46]
    }
    recent_df = pd.DataFrame(data)
    recent_df['recent_GF_pm'] = recent_df['recent_GF_total'] / recent_df['recent_form_matches']
    recent_df['recent_GA_pm'] = recent_df['recent_GA_total'] / recent_df['recent_form_matches']
    recent_df['recent_xG_pm'] = recent_df['recent_xG_total'] / recent_df['recent_xg_matches']
    recent_df['recent_xGA_pm'] = recent_df['recent_xGA_total'] / recent_df['recent_xg_matches']
    return recent_df

def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0:
        return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1:
        return 1 + (l_h * rho)
    if x == 1 and y == 0:
        return 1 + (m_a * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1

def get_recent_bonus_df(league_name):
    if league_name == "Bundesliga":
        return load_bundesliga_recent_bonus_data()
    if league_name == "Premier League":
        return load_premier_league_recent_bonus_data()
    if league_name == "La Liga":
        return load_la_liga_recent_bonus_data()
    return None

def calculate_recent_bonus(team_name, base_row, recent_df):
    eps = 1e-9
    team_recent = recent_df[recent_df['Team'] == team_name]

    if team_recent.empty:
        return {
            "team": team_name,
            "season_GF": float(base_row['T_GF']),
            "season_GA": float(base_row['T_GA']),
            "season_xG": float(base_row['TxG_F']),
            "season_xGA": float(base_row['TxG_A']),
            "recent_GF_pm": float(base_row['T_GF']),
            "recent_GA_pm": float(base_row['T_GA']),
            "recent_xG_pm": float(base_row['TxG_F']),
            "recent_xGA_pm": float(base_row['TxG_A']),
            "recent_form_matches": 0,
            "recent_xg_matches": 0,
            "trend_creation": 0.0,
            "finishing": 0.0,
            "raw_attack_bonus": 0.0,
            "attack_bonus": 0.0,
            "trend_defense": 0.0,
            "defense_gk": 0.0,
            "raw_defense_bonus": 0.0,
            "defense_bonus": 0.0
        }

    r = team_recent.iloc[0]

    season_GF = float(base_row['T_GF'])
    season_GA = float(base_row['T_GA'])
    season_xG = float(base_row['TxG_F'])
    season_xGA = float(base_row['TxG_A'])

    recent_GF_pm = float(r['recent_GF_pm'])
    recent_GA_pm = float(r['recent_GA_pm'])
    recent_xG_pm = float(r['recent_xG_pm'])
    recent_xGA_pm = float(r['recent_xGA_pm'])

    trend_creation = (recent_xG_pm - season_xG) / max(season_xG, eps)
    finishing = (recent_GF_pm - recent_xG_pm) / max(recent_xG_pm, eps)
    raw_attack_bonus = (trend_creation * 0.7) + (finishing * 0.3)
    attack_bonus = raw_attack_bonus * 0.25

    trend_defense = (season_xGA - recent_xGA_pm) / max(season_xGA, eps)
    defense_gk = (recent_xGA_pm - recent_GA_pm) / max(recent_xGA_pm, eps)
    raw_defense_bonus = (trend_defense * 0.7) + (defense_gk * 0.3)
    defense_bonus = raw_defense_bonus * 0.25

    return {
        "team": team_name,
        "season_GF": season_GF,
        "season_GA": season_GA,
        "season_xG": season_xG,
        "season_xGA": season_xGA,
        "recent_GF_pm": recent_GF_pm,
        "recent_GA_pm": recent_GA_pm,
        "recent_xG_pm": recent_xG_pm,
        "recent_xGA_pm": recent_xGA_pm,
        "recent_form_matches": int(r['recent_form_matches']),
        "recent_xg_matches": int(r['recent_xg_matches']),
        "trend_creation": trend_creation,
        "finishing": finishing,
        "raw_attack_bonus": raw_attack_bonus,
        "attack_bonus": attack_bonus,
        "trend_defense": trend_defense,
        "defense_gk": defense_gk,
        "raw_defense_bonus": raw_defense_bonus,
        "defense_bonus": defense_bonus
    }

def render_recent_bonus_table(bonus):
    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:20px;">
        <tr>
            <th style="border:1px solid #bbb; padding:10px; text-align:left; background:#f2f2f2;">Parametr (na mecz)</th>
            <th style="border:1px solid #bbb; padding:10px; text-align:left; background:#f2f2f2;">Sezon (Baza)</th>
            <th style="border:1px solid #bbb; padding:10px; text-align:left; background:#f2f2f2;">Ostatnie mecze</th>
        </tr>
        <tr>
            <td style="border:1px solid #bbb; padding:10px;"><b>Gole Strzelone (GF)</b></td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;">{bonus['season_GF']:.2f}</td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;"><b>{bonus['recent_GF_pm']:.2f}</b> <span style="font-size:13px;">(ostatnie {bonus['recent_form_matches']} meczów)</span></td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb; padding:10px;"><b>Gole Stracone (GA)</b></td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;">{bonus['season_GA']:.2f}</td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;"><b>{bonus['recent_GA_pm']:.2f}</b> <span style="font-size:13px;">(ostatnie {bonus['recent_form_matches']} meczów)</span></td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb; padding:10px;"><b>xG (Kreacja)</b></td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;">{bonus['season_xG']:.2f}</td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;"><b>{bonus['recent_xG_pm']:.2f}</b> <span style="font-size:13px;">(ostatnie {bonus['recent_xg_matches']} mecze)</span></td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb; padding:10px;"><b>xGA (Dopuszczone)</b></td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;">{bonus['season_xGA']:.2f}</td>
            <td style="border:1px solid #bbb; padding:10px; background:#fff275;"><b>{bonus['recent_xGA_pm']:.2f}</b> <span style="font-size:13px;">(ostatnie {bonus['recent_xg_matches']} mecze)</span></td>
        </tr>
    </table>
    """
    st.markdown(f"### Dane wejściowe dla {bonus['team']}")
    st.markdown(table_html, unsafe_allow_html=True)

def render_recent_bonus_details(bonus):
    atk_sign = "podniesiona" if bonus['attack_bonus'] >= 0 else "obniżona"
    def_sign = "lepiej" if bonus['defense_bonus'] >= 0 else "gorzej"

    st.markdown(f"## 2. Obliczanie Bonusu Ataku {bonus['team']}")
    st.markdown(
        f"""
- **Trend Kreacji (70%)**: ({bonus['recent_xG_pm']:.2f} - {bonus['season_xG']:.2f}) / {bonus['season_xG']:.2f} = **{bonus['trend_creation']:+.1%}**
- **Skuteczność (30%)**: ({bonus['recent_GF_pm']:.2f} - {bonus['recent_xG_pm']:.2f}) / {bonus['recent_xG_pm']:.2f} = **{bonus['finishing']:+.1%}**
- **Surowy Bonus Ataku**: ({bonus['trend_creation']:+.1%} × 0.7) + ({bonus['finishing']:+.1%} × 0.3) = **{bonus['raw_attack_bonus']:+.2%}**
- **Po tłumieniu (× 0.25)**: {bonus['raw_attack_bonus']:+.2%} × 0.25 = **{bonus['attack_bonus']:+.2%}**
        """
    )
    st.markdown(f"**Wynik:** Siła ataku **{bonus['team']}** zostaje **{atk_sign}** o **{abs(bonus['attack_bonus']):.2%}**.")

    st.markdown(f"## 3. Obliczanie Bonusu Obrony {bonus['team']}")
    st.markdown(f"Sprawdzamy, jak radzi sobie blok defensywny i bramkarz **{bonus['team']}**.")
    st.markdown(
        f"""
- **Trend Defensywny (70%)**: ({bonus['season_xGA']:.2f} - {bonus['recent_xGA_pm']:.2f}) / {bonus['season_xGA']:.2f} = **{bonus['trend_defense']:+.1%}**
- **Skuteczność Obrony/GK (30%)**: ({bonus['recent_xGA_pm']:.2f} - {bonus['recent_GA_pm']:.2f}) / {bonus['recent_xGA_pm']:.2f} = **{bonus['defense_gk']:+.1%}**
- **Surowy Bonus Obrony**: ({bonus['trend_defense']:+.1%} × 0.7) + ({bonus['defense_gk']:+.1%} × 0.3) = **{bonus['raw_defense_bonus']:+.2%}**
- **Po tłumieniu (× 0.25)**: {bonus['raw_defense_bonus']:+.2%} × 0.25 = **{bonus['defense_bonus']:+.2%}**
        """
    )
    st.markdown(f"**Wynik:** Obrona **{bonus['team']}** jest oceniana o **{abs(bonus['defense_bonus']):.2%} {def_sign}** niż średnia z sezonu.")

if 'mod_reset' not in st.session_state:
    st.session_state.mod_reset = 0

def reset_mods():
    st.session_state.mod_reset += 1

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def reset_weights():
    st.session_state.reset_counter += 1

st.sidebar.header("⚙️ Konfiguracja Wag")
st.sidebar.button("🔄 Resetuj wagi", on_click=reset_weights)

options = [i for i in range(0, 105, 5)]
v0 = st.sidebar.selectbox("🎯 xG Sezon D/W %", options, index=options.index(40), key=f"w0_{st.session_state.reset_counter}")
v1 = st.sidebar.selectbox("⚽ Gole Sezon D/W %", options, index=options.index(25), key=f"w1_{st.session_state.reset_counter}")
v2 = st.sidebar.selectbox("📊 xG Cały Sezon %", options, index=options.index(20), key=f"w2_{st.session_state.reset_counter}")
v3 = st.sidebar.selectbox("📉 Gole Cały Sezon %", options, index=options.index(15), key=f"w3_{st.session_state.reset_counter}")

if v0 + v1 + v2 + v3 != 100:
    st.sidebar.error("Suma wag musi wynosić 100%!")
    st.stop()

w0, w1, w2, w3 = v0 / 100, v1 / 100, v2 / 100, v3 / 100
fixed_rho = -0.15

tab_bl, tab_pl, tab_ll = st.tabs(["🇩🇪 Bundesliga", "🏴 Premier League", "🇪🇸 La Liga"])

def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    st.title(f"⚽ {league_name} Predictor")

    col_a, col_b = st.columns(2)

    with col_a:
        h_team = st.selectbox("Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
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
        a_team = st.selectbox("Gość", df['Team'], index=1, key=f"a_{league_name}")
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

    h = df[df['Team'] == h_team].iloc[0]
    a = df[df['Team'] == a_team].iloc[0]

    h_recent_attack_bonus = 0.0
    h_recent_defense_bonus = 0.0
    a_recent_attack_bonus = 0.0
    a_recent_defense_bonus = 0.0

    recent_df = get_recent_bonus_df(league_name)

    if recent_df is not None:
        st.divider()
        st.markdown("## 🔥 Bonusy za ostatnie mecze")

        h_bonus = calculate_recent_bonus(h_team, h, recent_df)
        a_bonus = calculate_recent_bonus(a_team, a, recent_df)

        h_recent_attack_bonus = h_bonus["attack_bonus"]
        h_recent_defense_bonus = h_bonus["defense_bonus"]
        a_recent_attack_bonus = a_bonus["attack_bonus"]
        a_recent_defense_bonus = a_bonus["defense_bonus"]

        bonus_col1, bonus_col2 = st.columns(2)
        with bonus_col1:
            render_recent_bonus_table(h_bonus)
            render_recent_bonus_details(h_bonus)
        with bonus_col2:
            render_recent_bonus_table(a_bonus)
            render_recent_bonus_details(a_bonus)

    l_h_r = (h['HxG_F'] * w0 + h['H_GF'] * w1 + h['AxG_F'] * w2 + h['T_GF'] * w3)
    m_h_r = (h['HxG_A'] * w0 + h['H_GA'] * w1 + h['AxG_A'] * w2 + h['T_GA'] * w3)
    l_a_r = (a['TxG_F'] * w0 + a['A_GF'] * w1 + a['AxG_F'] * w2 + a['T_GF'] * w3)
    m_a_r = (a['TxG_A'] * w0 + a['A_GA'] * w1 + a['AxG_A'] * w2 + a['T_GA'] * w3)

    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)

    lambda_base = (h_atk_s * a_def_s * avg_h_gf) * (1 + h_total_mod)
    mu_base = (a_atk_s * h_def_s * avg_a_gf) * (1 + a_total_mod)

    if recent_df is not None:
        home_recent_multiplier = max((1 + h_recent_attack_bonus) * (1 - a_recent_defense_bonus), 0.50)
        away_recent_multiplier = max((1 + a_recent_attack_bonus) * (1 - h_recent_defense_bonus), 0.50)
        lambda_f = lambda_base * home_recent_multiplier
        mu_f = mu_base * away_recent_multiplier
    else:
        home_recent_multiplier = 1.0
        away_recent_multiplier = 1.0
        lambda_f = lambda_base
        mu_f = mu_base

    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, fixed_rho)
    matrix /= matrix.sum()

    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))
    model_odds = [1 / max(p1, 0.001), 1 / max(px, 0.001), 1 / max(p2, 0.001)]

    st.divider()

    if recent_df is not None:
        st.markdown("### 🧩 Wpływ bonusów ostatnich meczów na lambdy")
        adj1, adj2 = st.columns(2)
        with adj1:
            st.metric(f"Lambda {h_team}", f"{lambda_f:.3f}", f"Base: {lambda_base:.3f} | Mnożnik formy: {home_recent_multiplier:.3f}")
        with adj2:
            st.metric(f"Lambda {a_team}", f"{mu_f:.3f}", f"Base: {mu_base:.3f} | Mnożnik formy: {away_recent_multiplier:.3f}")

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
            gf, ga, xgf, xga = team_data['T_GF'], team_data['T_GA'], team_data['TxG_F'], team_data['TxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['T_GF'].mean(), full_df['T_GA'].mean(), full_df['TxG_F'].mean(), full_df['TxG_A'].mean()
        elif context == "Dom":
            gf, ga, xgf, xga = team_data['H_GF'], team_data['H_GA'], team_data['HxG_F'], team_data['HxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['H_GF'].mean(), full_df['H_GA'].mean(), full_df['HxG_F'].mean(), full_df['HxG_A'].mean()
        else:
            gf, ga, xgf, xga = team_data['A_GF'], team_data['A_GA'], team_data['AxG_F'], team_data['AxG_A']
            l_avg_gf, l_avg_ga, l_avg_xgf, l_avg_xga = full_df['A_GF'].mean(), full_df['A_GA'].mean(), full_df['AxG_F'].mean(), full_df['AxG_A'].mean()

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
    | **Łączny Modyfikator** | **{h_total_mod:+.0%}** | **{a_total_mod:+.0%}** |
    """)

    with st.expander("🧮 Szczegółowa Ścieżka Obliczeniowa"):
        st.subheader("1. Średnie ligowe")
        st.write(f"Średnia gospodarzy: `{avg_h_gf:.3f}` | Średnia gości: `{avg_a_gf:.3f}`")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**{h_team}**")
            st.write(f"🎯 **Bazowa Siła Ataku:** `{l_h_r:.3f} / {avg_h_gf:.3f} = {h_atk_s:.3f}`")
            if recent_df is not None:
                st.write(f"🔥 **Bonus ataku ostatnich meczów:** `{h_recent_attack_bonus:+.2%}`")
                st.write(f"🛡️ **Bonus obrony rywala:** `{a_recent_defense_bonus:+.2%}`")
                st.write(f"⚙️ **Mnożnik formy:** `{home_recent_multiplier:.3f}`")
        with sc2:
            st.markdown(f"**{a_team}**")
            st.write(f"🎯 **Bazowa Siła Ataku:** `{l_a_r:.3f} / {avg_a_gf:.3f} = {a_atk_s:.3f}`")
            if recent_df is not None:
                st.write(f"🔥 **Bonus ataku ostatnich meczów:** `{a_recent_attack_bonus:+.2%}`")
                st.write(f"🛡️ **Bonus obrony rywala:** `{h_recent_defense_bonus:+.2%}`")
                st.write(f"⚙️ **Mnożnik formy:** `{away_recent_multiplier:.3f}`")

        st.subheader("2. Parametry Poisson (Skorygowane)")
        if recent_df is not None:
            st.latex(rf"\lambda_{{base}} = {lambda_base:.3f}")
            st.latex(rf"\lambda_{{final}} = \lambda_{{base}} \times {home_recent_multiplier:.3f} = {lambda_f:.3f}")
            st.latex(rf"\mu_{{base}} = {mu_base:.3f}")
            st.latex(rf"\mu_{{final}} = \mu_{{base}} \times {away_recent_multiplier:.3f} = {mu_f:.3f}")

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
            st.write(f"🟢 **OVER**: {prob_over:.1%} (Kurs: {1 / max(prob_over, 0.001):.2f})")
            st.write(f"🔴 **UNDER**: {prob_under:.1%} (Kurs: {1 / max(prob_under, 0.001):.2f})")

    st.divider()
    st.subheader("🥅 Obie Drużyny Strzelą (BTTS)")
    prob_btts_yes = sum(matrix[x, y] for x in range(1, max_g) for y in range(1, max_g))
    prob_btts_no = 1 - prob_btts_yes
    b1, b2 = st.columns(2)
    with b1:
        st.write(f"🟢 **TAK**: {prob_btts_yes:.1%} (Kurs: {1 / max(prob_btts_yes, 0.001):.2f})")
    with b2:
        st.write(f"🔴 **NIE**: {prob_btts_no:.1%} (Kurs: {1 / max(prob_btts_no, 0.001):.2f})")

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
        if f"messages_{league_name}" not in st.session_state:
            st.session_state[f"messages_{league_name}"] = []
        for message in st.session_state[f"messages_{league_name}"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Zadaj pytanie...", key=f"chat_input_{league_name}"):
            st.session_state[f"messages_{league_name}"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=[
                            {"role": "system", "content": f"Ekspert piłkarski. Kontekst: {current_context}"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500
                    )
                    full_response = response.choices[0].message.content
                    placeholder.markdown(full_response)
                    st.session_state[f"messages_{league_name}"].append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Błąd AI: {str(e)}")
    else:
        st.info("Dodaj HF_TOKEN do Secrets.")

with tab_bl:
    render_league_ui(load_bundesliga(), "Bundesliga")

with tab_pl:
    render_league_ui(load_premier_league(), "Premier League")

with tab_ll:
    render_league_ui(load_la_liga(), "La Liga")
