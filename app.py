# ... (zachowaj poprzedni kod aż do linii mu_f = a_atk_s * h_def_s * avg_a_gf)

# --- NOWY MODUŁ: ŚCIEŻKA OBLICZENIOWA ---
with st.expander("🔍 Zobacz jak powstał wynik (Metodologia)"):
    st.subheader("1. Średnie ważone parametrów")
    st.write("Łączymy xG oraz realne gole zgodnie z Twoimi wagami:")
    
    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        st.markdown(f"**{h_team} (Dom)**")
        st.latex(rf"L_{{h}} = {h['HxG_F']:.2f} \cdot {w0} + {h['H_GF']:.2f} \cdot {w1} + {h['TxG_F']:.2f} \cdot {w2} + {h['T_GF']:.2f} \cdot {w3} = \mathbf{{{l_h_r:.2f}}}")
        st.latex(rf"M_{{h}} = {h['HxG_A']:.2f} \cdot {w0} + {h['H_GA']:.2f} \cdot {w1} + {h['TxG_A']:.2f} \cdot {w2} + {h['T_GA']:.2f} \cdot {w3} = \mathbf{{{m_h_r:.2f}}}")
    
    with calc_col2:
        st.markdown(f"**{a_team} (Wyjazd)**")
        st.latex(rf"L_{{a}} = {a['AxG_F']:.2f} \cdot {w0} + {a['A_GF']:.2f} \cdot {w1} + {a['TxG_F']:.2f} \cdot {w2} + {a['T_GF']:.2f} \cdot {w3} = \mathbf{{{l_a_r:.2f}}}")
        st.latex(rf"M_{{a}} = {a['AxG_A']:.2f} \cdot {w0} + {a['A_GA']:.2f} \cdot {w1} + {a['TxG_A']:.2f} \cdot {w2} + {a['T_GA']:.2f} \cdot {w3} = \mathbf{{{m_a_r:.2f}}}")

    st.subheader("2. Współczynniki Siły (Relative Strength)")
    st.write("Dzielimy wyniki przez średnią ligową (Dom: {:.2f}, Wyjazd: {:.2f}):".format(avg_h_gf, avg_a_gf))
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.info(f"Atak {h_team}: {l_h_r:.2f} / {avg_h_gf:.2f} = **{h_atk_s:.2f}**")
        st.info(f"Obrona {h_team}: {m_h_r:.2f} / {avg_a_gf:.2f} = **{h_def_s:.2f}**")
    with s_col2:
        st.info(f"Atak {a_team}: {l_a_r:.2f} / {avg_a_gf:.2f} = **{a_atk_s:.2f}**")
        st.info(f"Obrona {a_team}: {m_a_r:.2f} / {avg_h_gf:.2f} = **{a_def_s:.2f}**")

    st.subheader("3. Finalna Prognoza Goli (Expected Goals)")
    st.write("Mnożymy siłę ataku jednej drużyny przez siłę obrony drugiej i średnią przewagę własnego boiska:")
    
    f_col1, f_col2 = st.columns(2)
    f_col1.success(f"λ (Gospodarz): {h_atk_s:.2f} * {a_def_s:.2f} * {avg_h_gf:.2f} = **{lambda_f:.2f} gola**")
    f_col2.success(f"μ (Gość): {a_atk_s:.2f} * {h_def_s:.2f} * {avg_a_gf:.2f} = **{mu_f:.2f} gola**")

# --- KONTYNUACJA ISTNIEJĄCEGO KODU (WIDOK 1X2) ---
# ...
