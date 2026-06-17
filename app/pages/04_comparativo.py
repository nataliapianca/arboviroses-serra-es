# -*- coding: utf-8 -*-
"""
Doenças Tropicais — Serra/ES
Comparativo & Dashboard Final — Entrega 2 PI-III
Pessoa 5 · Natalia Pianca
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Comparativo — Serra/ES",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=400;600&family=IBM+Plex+Sans:wght=300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Fundo claro e texto principal escuro */
.stApp { background: #ffffff; color: #1f2328; }

h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #1f2328 !important;
    letter-spacing: -0.02em;
}

[data-testid="metric-container"] {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #57606a !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 28px !important;
    color: #1f2328 !important;
}

div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px;
    color: #57606a;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0969da !important;
    border-bottom-color: #0969da !important;
}

.stSelectbox > div > div, .stSlider { background: transparent !important; }

hr { border-color: #d0d7de !important; }

/* Ajuste de contraste das caixas de alerta/informação */
.info-box {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-left: 3px solid #0969da;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #24292f;
    margin-bottom: 1rem;
}
.result-box {
    background: #f0fdf4;
    border: 1px solid #2da44e;
    border-left: 3px solid #1a7f37;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #1a7f37;
    margin-bottom: 1rem;
}
.warn-box {
    background: #fff8ec;
    border: 1px solid #bf8700;
    border-left: 3px solid #9a6700;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #9a6700;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
DOENCAS     = ["dengue", "zika", "chikungunya"]
CORES       = {"dengue": "#f85149", "zika": "#3fb950", "chikungunya": "#d29922"}
NOMES_MES   = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

# Caminhos dos CSVs gerados pelo notebook
BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_MENSAL  = os.path.join(BASE, "data", "comparativo", "comparativo_mensal.csv")
CSV_ANUAL   = os.path.join(BASE, "data", "comparativo", "comparativo_anual.csv")
CSV_RESUMO  = os.path.join(BASE, "data", "comparativo", "resumo_comparativo.csv")

# ─────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Carrega CSVs gerados pelo notebook. Fallback simulado se ausentes."""
    try:
        casos   = pd.read_csv(CSV_MENSAL,  parse_dates=["data"])
        anual   = pd.read_csv(CSV_ANUAL)
        resumo  = pd.read_csv(CSV_RESUMO)
        casos["ano"] = casos["data"].dt.year
        casos["mes"] = casos["data"].dt.month
        fonte = "InfoDengue/Fiocruz – Serra/ES"
        return casos, anual, resumo, fonte
    except FileNotFoundError:
        # ── fallback simulado ─────────────────────────────────
        np.random.seed(42)
        idx   = pd.date_range("2015-01", periods=9*12, freq="MS")
        t     = np.arange(len(idx))
        sazon = np.sin(2*np.pi*(t % 12)/12 - np.pi/2)*0.5 + 0.5
        casos = pd.DataFrame({
            "data":        idx,
            "dengue":      (sazon*400 + np.random.randint(50, 150, len(idx))).astype(int),
            "zika":        (sazon*100 + np.random.randint(10,  40, len(idx))).astype(int),
            "chikungunya": (sazon*60  + np.random.randint(5,   25, len(idx))).astype(int),
        })
        casos["ano"] = casos["data"].dt.year
        casos["mes"] = casos["data"].dt.month
        anual  = casos.groupby("ano")[DOENCAS].sum().reset_index()
        resumo_rows = []
        for d in DOENCAS:
            media_mes = casos.groupby("mes")[d].mean()
            resumo_rows.append({
                "Doença":          d.capitalize(),
                "Total (2015–23)": int(casos[d].sum()),
                "Média mensal":    round(casos[d].mean(), 1),
                "Máximo mensal":   int(casos[d].max()),
                "Mês de pico":     NOMES_MES[media_mes.idxmax() - 1],
                "Ano de pico":     int(casos.groupby("ano")[d].sum().idxmax()),
            })
        resumo = pd.DataFrame(resumo_rows)
        fonte  = "⚠️ DADOS SIMULADOS (CSVs não encontrados)"
        return casos, anual, resumo, fonte


casos, df_anual, df_resumo, FONTE = load_data()


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("📊 Comparativo de Arboviroses — Serra/ES")
st.caption(f"Dengue · Zika · Chikungunya · 2015–2023 · Fonte: {FONTE}")
st.divider()

# ─────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Dengue",       f"{casos['dengue'].sum():,}")
c2.metric("Total Zika",         f"{casos['zika'].sum():,}")
c3.metric("Total Chikungunya",  f"{casos['chikungunya'].sum():,}")
total_geral = casos[DOENCAS].sum().sum()
pct_dengue  = casos["dengue"].sum() / total_geral * 100
c4.metric("Dengue / total",     f"{pct_dengue:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Série temporal",
    "📅  Total anual",
    "🌀  Sazonalidade & Tendência",
    "📋  Tabela & Download",
])


# ══════════════════════════════════════════════════════════════
# ABA 1 — SÉRIE TEMPORAL COMPARATIVA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Notificações mensais — 3 doenças")
    st.markdown('<div class="info-box">Série mensal de casos notificados para as três arboviroses no mesmo eixo.</div>', unsafe_allow_html=True)

    fig_serie = go.Figure()
    for d in DOENCAS:
        fig_serie.add_trace(go.Scatter(
            x=casos["data"], y=casos[d],
            name=d.capitalize(),
            line=dict(color=CORES[d], width=2),
            mode="lines",
            hovertemplate="%{x|%b/%Y}<br>Casos: %{y}<extra>" + d.capitalize() + "</extra>",
        ))
    fig_serie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        margin=dict(t=30, b=0),
        height=420,
    )
    st.plotly_chart(fig_serie, use_container_width=True)

    st.divider()

    # ── Proporção em área empilhada ───────────────────────────
    st.subheader("Proporção relativa entre doenças (%)")

    df_prop = casos[["data"] + DOENCAS].copy()
    total_m = df_prop[DOENCAS].sum(axis=1).replace(0, np.nan)
    for d in DOENCAS:
        df_prop[d] = df_prop[d] / total_m * 100

    fig_area = go.Figure()
    for d in DOENCAS:
        fig_area.add_trace(go.Scatter(
            x=df_prop["data"], y=df_prop[d],
            name=d.capitalize(),
            mode="lines",
            stackgroup="one",
            line=dict(color=CORES[d], width=0.5),
            fillcolor=CORES[d],
        ))
    fig_area.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="% do total de casos",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        height=350,
    )
    st.plotly_chart(fig_area, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 2 — TOTAL ANUAL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Total anual por doença")

    fig_bar = go.Figure()
    for d in DOENCAS:
        fig_bar.add_trace(go.Bar(
            x=df_anual["ano"],
            y=df_anual[d],
            name=d.capitalize(),
            marker_color=CORES[d],
        ))
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        xaxis_title="Ano",
        yaxis_title="Total de casos",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Barras empilhadas ─────────────────────────────────────
    st.subheader("Composição anual (empilhado)")

    fig_stack = go.Figure()
    for d in DOENCAS:
        fig_stack.add_trace(go.Bar(
            x=df_anual["ano"],
            y=df_anual[d],
            name=d.capitalize(),
            marker_color=CORES[d],
        ))
    fig_stack.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        xaxis_title="Ano",
        yaxis_title="Total de casos",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        height=380,
    )
    st.plotly_chart(fig_stack, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 3 — SAZONALIDADE & TENDÊNCIA
# ══════════════════════════════════════════════════════════════
with tab3:

    # ── Sazonalidade ──────────────────────────────────────────
    st.subheader("Padrão sazonal médio por mês")
    st.markdown('<div class="info-box">Média histórica de casos por mês (2015–2023) — todas as doenças.</div>', unsafe_allow_html=True)

    fig_sazon = go.Figure()
    for d in DOENCAS:
        media_mes = casos.groupby("mes")[d].mean().values
        fig_sazon.add_trace(go.Scatter(
            x=NOMES_MES, y=media_mes,
            name=d.capitalize(),
            line=dict(color=CORES[d], width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
        ))
    fig_sazon.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mês",
        yaxis_title="Média de casos",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        height=380,
    )
    st.plotly_chart(fig_sazon, use_container_width=True)

    # Mês de pico por doença
    pk1, pk2, pk3 = st.columns(3)
    for col, d in zip([pk1, pk2, pk3], DOENCAS):
        media_mes = casos.groupby("mes")[d].mean()
        pico = NOMES_MES[media_mes.idxmax() - 1]
        col.metric(f"Pico — {d.capitalize()}", pico)

    st.divider()

    # ── Tendências ────────────────────────────────────────────
    st.subheader("Tendências extraídas por decomposição")
    st.markdown('<div class="info-box">Componente de tendência da decomposição aditiva (statsmodels, período=12).</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    for d in DOENCAS:
        try:
            serie = casos.set_index("data")[d].asfreq("MS")
            dec   = seasonal_decompose(serie, model="additive", period=12, extrapolate_trend="freq")
            fig_trend.add_trace(go.Scatter(
                x=dec.trend.index, y=dec.trend.values,
                name=f"{d.capitalize()} — tendência",
                line=dict(color=CORES[d], width=2.5),
                mode="lines",
            ))
        except Exception:
            pass

    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Ano",
        yaxis_title="Casos (componente tendência)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 4 — TABELA & DOWNLOAD
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Tabela-resumo comparativa")
    st.markdown('<div class="info-box">Total, média mensal, máximo, mês e ano de pico — por doença.</div>', unsafe_allow_html=True)

    st.dataframe(df_resumo, use_container_width=True, hide_index=True)

    st.divider()

    # ── Downloads ─────────────────────────────────────────────
    st.subheader("Download dos dados")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        csv_mensal = casos[["data"] + DOENCAS].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Série mensal (CSV)",
            data=csv_mensal,
            file_name="comparativo_mensal.csv",
            mime="text/csv",
        )

    with col_d2:
        csv_anual = df_anual.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Total anual (CSV)",
            data=csv_anual,
            file_name="comparativo_anual.csv",
            mime="text/csv",
        )

    with col_d3:
        csv_resumo = df_resumo.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Tabela-resumo (CSV)",
            data=csv_resumo,
            file_name="resumo_comparativo.csv",
            mime="text/csv",
        )

    st.divider()

    # ── Série mensal completa ─────────────────────────────────
    st.subheader("Série mensal completa")
    st.dataframe(
        casos[["data"] + DOENCAS].sort_values("data", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=400,
    )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption(f"FAESA · Projeto Integrador III · Ciência de Dados · 2025 · Fonte: {FONTE}")