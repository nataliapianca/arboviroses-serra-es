# -*- coding: utf-8 -*-
"""
Doenças Tropicais — Serra/ES
Dashboard epidemiológico — Entrega 2 PI-III
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import os
 
# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Doenças Tropicais — Serra/ES",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
 
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
 
.stApp { background: #0d1117; color: #e6edf3; }
 
h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #e6edf3 !important;
    letter-spacing: -0.02em;
}
 
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 28px !important;
    color: #e6edf3 !important;
}
 
div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px;
    color: #8b949e;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}
 
.stSelectbox > div > div, .stSlider {
    background: transparent !important;
}
 
hr { border-color: #21262d !important; }
 
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 1rem;
}
 
.corr-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
}
.corr-table th {
    background: #161b22;
    color: #8b949e;
    padding: 8px 14px;
    text-align: left;
    border-bottom: 1px solid #21262d;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: .06em;
}
.corr-table td {
    padding: 8px 14px;
    border-bottom: 1px solid #21262d;
    color: #e6edf3;
}
.corr-table tr:last-child td { border-bottom: none; }
.pos { color: #3fb950; }
.neg { color: #f85149; }
.neutral { color: #8b949e; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────
 
DOENCAS = ["dengue", "zika", "chikungunya"]
CORES   = {"dengue": "#f85149", "zika": "#3fb950", "chikungunya": "#d29922"}
 
BAIRROS = [
    "Jardim Carapina", "Laranjeiras", "Serra Sede", "Novo Horizonte",
    "Parque Residencial Laranjeiras", "Barcelona", "Colina de Laranjeiras",
    "Civit II", "Carapina Grande", "Taquara I", "Portal de Jacaraípe",
    "Eldorado", "Manguinhos", "São Marcos", "Vila Nova de Colares",
]
 
POPULACAO_BAIRROS = {b: np.random.randint(8000, 50000) for b in BAIRROS}
 
 
@st.cache_data
def load_data():
    """
    Carrega dados do projeto.
    ─────────────────────────────────────────────────────────
    PARA USAR DADOS REAIS: substitua este bloco por:
 
        casos  = pd.read_csv("dados/casos_limpos.csv", parse_dates=["data"])
        decomp = pd.read_csv("dados/decomposicao.csv", parse_dates=["data"])
        corr   = pd.read_csv("dados/correlacoes.csv")
        bairros= pd.read_csv("dados/incidencia_bairros.csv")
        return casos, decomp, corr, bairros
 
    Os CSVs devem ser gerados pelos notebooks de análise.
    ─────────────────────────────────────────────────────────
    """
    np.random.seed(42)
    datas = pd.date_range("2015-01-01", "2023-12-01", freq="MS")
    n = len(datas)
 
    # Sazonalidade realista (pico fev-mar para dengue)
    t = np.arange(n)
    sazon = np.sin(2 * np.pi * (t % 12) / 12 - np.pi / 2) * 0.5 + 0.5
 
    casos = pd.DataFrame({
        "data":        datas,
        "dengue":      (sazon * 400 + np.random.randint(50, 150, n)).astype(int),
        "zika":        (sazon * 100 + np.random.randint(10, 40,  n)).astype(int),
        "chikungunya": (sazon * 60  + np.random.randint(5,  25,  n)).astype(int),
        "temp":        np.clip(25 + sazon * 4 + np.random.normal(0, 0.8, n), 20, 35),
        "chuva":       np.clip(sazon * 200 + np.random.randint(30, 80, n), 20, 350),
    })
 
    # Dados por bairro
    bairros = pd.DataFrame({
        "bairro":    BAIRROS,
        "populacao": [POPULACAO_BAIRROS[b] for b in BAIRROS],
        "dengue":    np.random.randint(200, 2000, len(BAIRROS)),
        "zika":      np.random.randint(50,  500,  len(BAIRROS)),
        "chikungunya": np.random.randint(20, 300, len(BAIRROS)),
    })
    for d in DOENCAS:
        bairros[f"incidencia_{d}"] = (
            bairros[d] / bairros["populacao"] * 100_000
        ).round(1)
 
    return casos, bairros
 
 
casos, bairros = load_data()
 
 
# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🦟 Doenças Tropicais — Serra/ES")
st.caption("Dashboard epidemiológico · Dengue · Zika · Chikungunya · 2015–2023")
st.divider()
 
# ─────────────────────────────────────────────────────────────
# FILTROS GLOBAIS
# ─────────────────────────────────────────────────────────────
colF1, colF2 = st.columns([2, 2])
 
with colF1:
    doenca = st.selectbox("Doença selecionada", DOENCAS,
                          format_func=lambda x: x.capitalize())
 
with colF2:
    anos = st.slider("Período", 2015, 2023, (2015, 2023))
 
df = casos[
    (casos["data"].dt.year >= anos[0]) &
    (casos["data"].dt.year <= anos[1])
].copy()
 
st.divider()
 
# ─────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Visão geral",
    "📈  Séries temporais",
    "🌡️  Correlação climática",
    "🗺️  Análise espacial",
])
 
 
# ══════════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
with tab1:
 
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dengue",       f"{df['dengue'].sum():,}")
    c2.metric("Total Zika",         f"{df['zika'].sum():,}")
    c3.metric("Total Chikungunya",  f"{df['chikungunya'].sum():,}")
    ano_pico = df.groupby(df["data"].dt.year)[doenca].sum().idxmax()
    c4.metric("Ano de pico", str(ano_pico))
 
    st.divider()
 
    # Evolução temporal — todas as doenças
    st.subheader("Evolução mensal — todas as doenças")
 
    fig_all = go.Figure()
    for d, cor in CORES.items():
        fig_all.add_trace(go.Scatter(
            x=df["data"], y=df[d],
            name=d.capitalize(),
            line=dict(color=cor, width=2),
            mode="lines",
        ))
    fig_all.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_all, use_container_width=True)
 
    # Anual
    st.subheader(f"Total anual — {doenca.capitalize()}")
    df["ano"] = df["data"].dt.year
    df_ano = df.groupby("ano")[doenca].sum().reset_index()
    fig_bar = px.bar(
        df_ano, x="ano", y=doenca,
        color_discrete_sequence=[CORES[doenca]],
        template="plotly_dark",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# ABA 2 — SÉRIES TEMPORAIS
# ══════════════════════════════════════════════════════════════
with tab2:
 
    st.subheader(f"Decomposição da série — {doenca.capitalize()}")
    st.markdown('<div class="info-box">Decomposição aditiva em tendência, sazonalidade e resíduo via statsmodels.</div>', unsafe_allow_html=True)
 
    serie = df.set_index("data")[doenca]
 
    try:
        decomp = seasonal_decompose(serie, model="additive", period=12, extrapolate_trend="freq")
 
        comp_cols = st.columns(3)
        componentes = {
            "Tendência":    decomp.trend,
            "Sazonalidade": decomp.seasonal,
            "Resíduo":      decomp.resid,
        }
        cores_comp = ["#58a6ff", "#3fb950", "#8b949e"]
 
        for i, (nome, comp) in enumerate(componentes.items()):
            with comp_cols[i]:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=comp.index, y=comp.values,
                    line=dict(color=cores_comp[i], width=1.8),
                    mode="lines", name=nome,
                ))
                fig_c.update_layout(
                    title=dict(text=nome, font=dict(size=13)),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40, b=0, l=0, r=0),
                    showlegend=False,
                    height=220,
                )
                st.plotly_chart(fig_c, use_container_width=True)
 
    except Exception as e:
        st.warning(f"Não foi possível decompor a série: {e}")
 
    st.divider()
 
    # Heatmap ano × mês
    st.subheader("Heatmap — ano × mês")
 
    df["mes"] = df["data"].dt.month
    pivot = df.pivot_table(index="ano", columns="mes", values=doenca, aggfunc="sum")
    pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun",
                     "Jul","Ago","Set","Out","Nov","Dez"]
 
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        template="plotly_dark",
        labels=dict(color="Casos"),
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
 
    st.divider()
 
    # Padrão sazonal médio
    st.subheader("Padrão sazonal médio por mês")
 
    meses_label = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    media_mes = df.groupby("mes")[doenca].mean().reset_index()
    media_mes["mes_label"] = meses_label
 
    mes_pico = media_mes.loc[media_mes[doenca].idxmax(), "mes_label"]
 
    fig_saz = px.bar(
        media_mes, x="mes_label", y=doenca,
        color_discrete_sequence=[CORES[doenca]],
        template="plotly_dark",
        labels={"mes_label": "Mês", doenca: "Média de casos"},
    )
    fig_saz.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=0),
    )
    st.plotly_chart(fig_saz, use_container_width=True)
    st.caption(f"📌 Mês de pico histórico: **{mes_pico}**")
 
 
# ══════════════════════════════════════════════════════════════
# ABA 3 — CORRELAÇÃO CLIMÁTICA
# ══════════════════════════════════════════════════════════════
with tab3:
 
    st.subheader("Correlação com variáveis climáticas")
    st.markdown('<div class="info-box">Coeficientes de Pearson e Spearman calculados entre casos mensais e temperatura/precipitação.</div>', unsafe_allow_html=True)
 
    # Calcula correlações para todas as doenças
    resultados = []
    for d in DOENCAS:
        for var, var_label in [("temp", "Temperatura"), ("chuva", "Precipitação")]:
            r_p, p_p = stats.pearsonr(df[d], df[var])
            r_s, p_s = stats.spearmanr(df[d], df[var])
            resultados.append({
                "Doença":      d.capitalize(),
                "Variável":    var_label,
                "Pearson r":   round(r_p, 3),
                "p (Pearson)": round(p_p, 4),
                "Spearman r":  round(r_s, 3),
                "p (Spearman)":round(p_s, 4),
            })
 
    df_corr = pd.DataFrame(resultados)
 
    # Tabela formatada
    def color_r(val):
        if val > 0.3:  return "pos"
        if val < -0.3: return "neg"
        return "neutral"
 
    rows_html = ""
    for _, row in df_corr.iterrows():
        cp = color_r(row["Pearson r"])
        cs = color_r(row["Spearman r"])
        rows_html += f"""
        <tr>
            <td>{row['Doença']}</td>
            <td>{row['Variável']}</td>
            <td class="{cp}">{row['Pearson r']}</td>
            <td style="color:#8b949e">{row['p (Pearson)']}</td>
            <td class="{cs}">{row['Spearman r']}</td>
            <td style="color:#8b949e">{row['p (Spearman)']}</td>
        </tr>"""
 
    st.markdown(f"""
    <table class="corr-table">
      <thead>
        <tr>
          <th>Doença</th><th>Variável</th>
          <th>Pearson r</th><th>p-valor</th>
          <th>Spearman r</th><th>p-valor</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
 
    st.caption("🟢 r > 0.3 (positivo)  |  🔴 r < -0.3 (negativo)  |  — sem correlação forte")
 
    st.divider()
 
    # Scatter plots
    st.subheader(f"Dispersão — {doenca.capitalize()}")
 
    sc1, sc2 = st.columns(2)
 
    with sc1:
        fig_t = px.scatter(
            df, x="temp", y=doenca,
            trendline="ols",
            color_discrete_sequence=[CORES[doenca]],
            template="plotly_dark",
            labels={"temp": "Temperatura (°C)", doenca: "Casos"},
            title="Temperatura vs Casos",
        )
        fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t, use_container_width=True)
 
    with sc2:
        fig_c = px.scatter(
            df, x="chuva", y=doenca,
            trendline="ols",
            color_discrete_sequence=[CORES[doenca]],
            template="plotly_dark",
            labels={"chuva": "Precipitação (mm)", doenca: "Casos"},
            title="Precipitação vs Casos",
        )
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)
 
    # Mês de pico por doença
    st.divider()
    st.subheader("Mês de pico histórico por doença")
 
    meses_label = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    df["mes"] = df["data"].dt.month
 
    pk1, pk2, pk3 = st.columns(3)
    for col, d in zip([pk1, pk2, pk3], DOENCAS):
        mm = df.groupby("mes")[d].mean()
        pico = meses_label[mm.idxmax() - 1]
        col.metric(f"Pico — {d.capitalize()}", pico)
 
 
# ══════════════════════════════════════════════════════════════
# ABA 4 — ANÁLISE ESPACIAL
# ══════════════════════════════════════════════════════════════
with tab4:
 
    st.subheader("Incidência por bairro — Serra/ES")
    st.markdown(f"""
    <div class="info-box">
        Taxa de incidência = casos / população × 100.000 hab.
        Dados simulados para fins acadêmicos.
        Para dados reais substitua <code>dados/incidencia_bairros.csv</code>.
    </div>
    """, unsafe_allow_html=True)
 
    col_inc = f"incidencia_{doenca}"
 
    # Ranking
    df_rank = bairros[["bairro", "populacao", doenca, col_inc]].sort_values(
        col_inc, ascending=False
    ).reset_index(drop=True)
    df_rank.index += 1
    df_rank.columns = ["Bairro", "População", "Casos", "Incidência (/100k)"]
 
    st.subheader(f"Ranking — {doenca.capitalize()}")
    st.dataframe(df_rank, use_container_width=True, height=420)
 
    st.divider()
 
    # Barras horizontais
    st.subheader("Top 10 bairros por incidência")
 
    top10 = df_rank.head(10)
    fig_rank = px.bar(
        top10,
        x="Incidência (/100k)",
        y="Bairro",
        orientation="h",
        color_discrete_sequence=[CORES[doenca]],
        template="plotly_dark",
    )
    fig_rank.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=0),
    )
    st.plotly_chart(fig_rank, use_container_width=True)
 
    st.divider()
 
    # Mapa
    st.subheader("Mapa coroplético")
    st.markdown("""
    <div class="info-box">
        ⚠️ Mapa aguardando shapefile dos bairros de Serra/ES.<br>
        Para ativar: adicione <code>dados/serra_bairros.shp</code> e descomente o bloco abaixo.
    </div>
    """, unsafe_allow_html=True)
 
    # ─── DESCOMENTE QUANDO TIVER O SHAPEFILE ───────────────────
    # import geopandas as gpd
    # import folium
    # from streamlit_folium import st_folium
    #
    # gdf = gpd.read_file("dados/serra_bairros.shp")
    # gdf = gdf.merge(bairros, on="bairro", how="left")
    #
    # m = folium.Map(location=[-20.13, -40.30], zoom_start=12, tiles="CartoDB dark_matter")
    # folium.Choropleth(
    #     geo_data=gdf,
    #     data=gdf,
    #     columns=["bairro", col_inc],
    #     key_on="feature.properties.bairro",
    #     fill_color="YlOrRd",
    #     fill_opacity=0.7,
    #     line_opacity=0.3,
    #     legend_name=f"Incidência de {doenca} (/100k hab.)",
    # ).add_to(m)
    # st_folium(m, use_container_width=True, height=500)
    # ───────────────────────────────────────────────────────────
 
    st.info("Mapa será exibido aqui após integração do shapefile dos bairros.")
 
 
# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("FAESA · Projeto Integrador III · Ciência de Dados · 2025 · dados simulados para fins acadêmicos")