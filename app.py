import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor & AI Chat", layout="wide", page_icon="⚽")

# =========================================================
# --- DANE (SKRÓCONE BO TWOJE BYŁY POPRAWNE) ---
# =========================================================

@st.cache_data
def load_bundesliga():
    data = {
        'Team': ['Bayern Munich','Borussia Dortmund','RB Leipzig','Bayer Leverkusen'],
        'H_GF':[4.00,2.33,2.25,2.08],
        'H_GA':[1.00,0.92,1.42,0.92],
        'A_GF':[3.33,1.92,1.58,1.67],
        'A_GA':[0.92,1.17,1.33,1.50],
        'HxG_F':[3.43,2.00,2.65,2.26],
        'HxG_A':[1.04,1.23,1.51,0.92],
        'AxG_F':[2.72,1.70,1.76,1.77],
        'AxG_A':[1.21,1.41,1.34,1.62],
        'T_GF':[3.67,2.13,1.92,1.88],
        'T_GA':[0.96,1.04,1.38,1.21],
        'TxG_F':[3.07,1.85,2.20,2.02],
        'TxG_A':[1.13,1.32,1.42,1.27],
        'Logo_ID':[27,16,23826,15]
    }
    return pd.DataFrame(data)

# =========================================================
# --- FUNKCJE MODELU ---
# =========================================================

def dixon_coles_adjustment(x, y, l_h, m_a, rho):
    if x == 0 and y == 0: return 1 - (l_h * m_a * rho)
    if x == 0 and y == 1: return 1 + (l_h * rho)
    if x == 1 and y == 0: return 1 + (m_a * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

# =========================================================
# --- SIDEBAR ---
# =========================================================

st.sidebar.header("⚙️ Wagi modelu")

options=[i for i in range(0,105,5)]

v0=st.sidebar.selectbox("xG dom/wyjazd",options,index=8)
v1=st.sidebar.selectbox("Gole dom/wyjazd",options,index=5)
v2=st.sidebar.selectbox("xG sezon",options,index=4)
v3=st.sidebar.selectbox("Gole sezon",options,index=3)

if v0+v1+v2+v3!=100:
    st.sidebar.error("Suma wag musi =100")
    st.stop()

w0,w1,w2,w3=v0/100,v1/100,v2/100,v3/100

# =========================================================
# --- INTERFEJS PREDYKTORA ---
# =========================================================

df=load_bundesliga()

st.title("⚽ Football Predictor")

col1,col2=st.columns(2)

with col1:
    home=st.selectbox("Gospodarz",df["Team"])

with col2:
    away=st.selectbox("Gość",df["Team"],index=1)

h=df[df.Team==home].iloc[0]
a=df[df.Team==away].iloc[0]

avg_h=df["H_GF"].mean()
avg_a=df["A_GF"].mean()

l_h=(h.HxG_F*w0+h.H_GF*w1+h.TxG_F*w2+h.T_GF*w3)
m_h=(h.HxG_A*w0+h.H_GA*w1+h.TxG_A*w2+h.T_GA*w3)

l_a=(a.AxG_F*w0+a.A_GF*w1+a.TxG_F*w2+a.T_GF*w3)
m_a=(a.AxG_A*w0+a.A_GA*w1+a.TxG_A*w2+a.T_GA*w3)

atk_h=l_h/avg_h
def_h=m_h/avg_a

atk_a=l_a/avg_a
def_a=m_a/avg_h

lambda_f=atk_h*def_a*avg_h
mu_f=atk_a*def_h*avg_a

max_g=10
matrix=np.zeros((max_g,max_g))

for x in range(max_g):
    for y in range(max_g):
        matrix[x,y]=poisson.pmf(x,lambda_f)*poisson.pmf(y,mu_f)

matrix/=matrix.sum()

p1=np.sum(np.tril(matrix,-1))
px=np.sum(np.diag(matrix))
p2=np.sum(np.triu(matrix,1))

c1,c2,c3=st.columns(3)

c1.metric(f"{home}",f"{p1:.1%}")
c2.metric("Remis",f"{px:.1%}")
c3.metric(f"{away}",f"{p2:.1%}")

st.write("### ExG")

e1,e2=st.columns(2)
e1.metric(home,f"{lambda_f:.2f}")
e2.metric(away,f"{mu_f:.2f}")

# =========================================================
# --- HEATMAP ---
# =========================================================

st.subheader("Macierz wyników")

fig,ax=plt.subplots()

sns.heatmap(matrix[:6,:6],annot=True,fmt=".1%",cbar=False)

st.pyplot(fig)

# =========================================================
# --- CHATBOT GEMINI ---
# =========================================================

st.markdown("---")
st.header("💬 AI Football Assistant")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Dodaj GOOGLE_API_KEY w Secrets")
    st.stop()

@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-2.0-flash")

model=load_model()

if "messages" not in st.session_state:
    st.session_state.messages=[]

with st.sidebar:

    st.divider()
    st.subheader("🤖 Chatbot")

    if st.button("Wyczyść czat"):
        st.session_state.messages=[]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt:=st.chat_input("Zapytaj o analizę meczu lub statystyki"):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder=st.empty()
        placeholder.markdown("*Analizuję...*")

        try:

            response=model.generate_content(prompt)

            if response.text:

                placeholder.markdown(response.text)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":response.text
                })

            else:
                placeholder.warning("Brak odpowiedzi")

        except Exception as e:

            err=str(e)

            if "404" in err:
                st.error("Model AI nie istnieje (404)")
            elif "429" in err:
                st.error("Limit zapytań API")
            else:
                st.error(err)
