def render_league_ui(df, league_name):
    avg_h_gf, avg_a_gf = df['H_GF'].mean(), df['A_GF'].mean()
    
    # --- NAGŁÓWEK Z HERBAMI ---
    st.markdown(f"<h1 style='text-align: center;'>⚽ {league_name} Predictor</h1>", unsafe_allow_html=True)
    
    col_a, col_spacer, col_b = st.columns([4, 1, 4])
    with col_a:
        h_team = st.selectbox(f"Gospodarz", df['Team'], index=0, key=f"h_{league_name}")
        h_id = df[df['Team'] == h_team]['Logo_ID'].values[0]
        st.markdown(f"<div style='text-align: center'><img src='https://tmssl.akamaized.net/images/wappen/head/{h_id}.png' width='120'></div>", unsafe_allow_html=True)
    
    with col_spacer:
        st.markdown("<h2 style='text-align: center; padding-top: 60px;'>VS</h2>", unsafe_allow_html=True)

    with col_b:
        a_team = st.selectbox(f"Gość", df['Team'], index=1, key=f"a_{league_name}")
        a_id = df[df['Team'] == a_team]['Logo_ID'].values[0]
        st.markdown(f"<div style='text-align: center'><img src='https://tmssl.akamaized.net/images/wappen/head/{a_id}.png' width='120'></div>", unsafe_allow_html=True)

    # Logika obliczeń (bez zmian)
    h, a = df[df['Team'] == h_team].iloc[0], df[df['Team'] == a_team].iloc[0]
    l_h_r = (h['HxG_F']*w0 + h['H_GF']*w1 + h['TxG_F']*w2 + h['T_GF']*w3)
    m_h_r = (h['HxG_A']*w0 + h['H_GA']*w1 + h['TxG_A']*w2 + h['T_GA']*w3)
    l_a_r = (a['AxG_F']*w0 + a['A_GF']*w1 + a['TxG_F']*w2 + a['T_GF']*w3)
    m_a_r = (a['AxG_A']*w0 + a['A_GA']*w1 + a['TxG_A']*w2 + a['T_GA']*w3)
    h_atk_s, h_def_s = (l_h_r / avg_h_gf), (m_h_r / avg_a_gf)
    a_atk_s, a_def_s = (l_a_r / avg_a_gf), (m_a_r / avg_h_gf)
    lambda_f = h_atk_s * a_def_s * avg_h_gf
    mu_f = a_atk_s * h_def_s * avg_a_gf

    max_g = 12
    matrix = np.zeros((max_g, max_g))
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lambda_f) * poisson.pmf(y, mu_f)
            matrix[x, y] = p * dixon_coles_adjustment(x, y, lambda_f, mu_f, fixed_rho)
    matrix /= matrix.sum()
    p1, px, p2 = np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

    # --- 1. PROCENTOWE SZANSE (Wizualne karty) ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.container(border=True).metric(f"🏠 {h_team}", f"{p1:.1%}", f"Kurs: {1/p1:.2f}")
    with c2:
        st.container(border=True).metric("🤝 Remis", f"{px:.1%}", f"Kurs: {1/px:.2f}")
    with c3:
        st.container(border=True).metric(f"🚀 {a_team}", f"{p2:.1%}", f"Kurs: {1/p2:.2f}")

    # --- 2. TABELA SIŁY (Nowoczesny Markdown) ---
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

    # --- 3. ŚCIEŻKA I MACIERZ (Schowane w Expanderach) ---
    with st.expander("🔍 Zobacz detale matematyczne i macierz goli"):
        col_mat1, col_mat2 = st.columns([1, 1])
        with col_mat1:
            st.latex(rf"\lambda = {lambda_f:.2f} \text{ (Gospodarz)}")
            st.latex(rf"\mu = {mu_f:.2f} \text{ (Gość)}")
        with col_mat2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(matrix[:6, :6], annot=True, fmt=".1%", cmap="Greens", cbar=False)
            plt.title("Prawdopodobieństwo Wyniku")
            st.pyplot(fig)

    # --- 4. UNDER / OVER & BTTS (Układ kafelkowy) ---
    st.markdown("### 🎯 Przewidywania Rynkowe")
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        with st.container(border=True):
            st.write("**Linia 2.5 Gola**")
            p_over25 = 1 - sum(matrix[x, y] for x in range(max_g) for y in range(max_g) if x + y < 2.5)
            st.write(f"🔥 Over: **{p_over25:.1%}** | ❄️ Under: **{1-p_over25:.1%}**")
    
    with row1_2:
        with st.container(border=True):
            st.write("**BTTS (Obie strzelą)**")
            p_btts = sum(matrix[x, y] for x in range(1, max_g) for y in range(1, max_g))
            st.write(f"✅ Tak: **{p_btts:.1%}** | ❌ Nie: **{1-p_btts:.1%}**")

    # --- 5. SYMULACJA MONTE CARLO (Bardziej interaktywna) ---
    st.divider()
    if st.button(f"🎲 SYMULUJ MECZ (10 000 prób)", use_container_width=True):
        with st.status("Trwa symulowanie spotkania...", expanded=True) as status:
            n_sim = 10000
            sim_h = np.random.poisson(lambda_f, n_sim)
            sim_a = np.random.poisson(mu_f, n_sim)
            
            res = pd.DataFrame({'H': sim_h, 'A': sim_a})
            res['Score'] = res['H'].astype(str) + ":" + res['A'].astype(str)
            most_common = res['Score'].value_counts().idxmax()
            
            st.write(f"🏆 Najczęstszy wynik w symulacji: **{most_common}**")
            
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            # Wykres rozkładu goli
            
            sns.kdeplot(sim_h, fill=True, color="#1f77b4", label=h_team, bw_adjust=3)
            sns.kdeplot(sim_a, fill=True, color="#ff7f0e", label=a_team, bw_adjust=3)
            plt.title("Gęstość prawdopodobieństwa zdobycia goli")
            plt.legend()
            st.pyplot(fig2)
            status.update(label="Symulacja zakończona!", state="complete", expanded=True)
