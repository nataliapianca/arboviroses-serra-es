import streamlit as st

pagina_serie_temporal = pagina_previsao = st.Page("pages/01_series_temporais.py", title="Séries Temporais", icon="📈")
pagina_previsao = st.Page("pages/02_previsao.py", title="Previsão", icon="📈")
pagina_mapa = st.Page("pages/03_mapa_bairros.py", title="Mapa dos Bairros", icon="🗺️", default=True)
pagina_comparativo = st.Page("pages/04_comparativo.py", title="Comparativo", icon="📊")

pagina_atual = st.navigation([pagina_serie_temporal,pagina_previsao, pagina_mapa, pagina_comparativo])
pagina_atual.run()