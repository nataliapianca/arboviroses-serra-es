# -*- coding: utf-8 -*-
"""
Séries Temporais & Sazonalidade — Serra/ES
Página 01 do Dashboard · Pessoa 2 — Séries Temporais & Sazonalidade
Mann-Kendall · Decomposição STL · Teste de Sazonalidade (Kruskal-Wallis)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import STL
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore") 

st.set_page_config(
    page_title="Séries Temporais — Serra/ES",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Alterado para fundo branco e letras escuras
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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
DOENCAS = ["dengue", "zika", "chikungunya"]
# Cores levemente ajustadas para melhor contraste em fundo claro
CORES   = {"dengue": "#cf222e", "zika": "#1a7f37", "chikungunya": "#9a6700"}
MESES_LABEL = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]


# ─────────────────────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Pichau\Documents\arboviroses-serra-es\serie-temporal\dados\comparativo_mensal.csv", parse_dates=["data"])
    except FileNotFoundError:
        st.error("⚠️ Arquivo dados/comparativo_mensal.csv não encontrado. Coloque o CSV de dados reais na pasta `dados/`.")
        st.stop()

    df = df.sort_values("data").reset_index(drop=True)
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    return df


@st.cache_data
def calcular_mann_kendall(df):
    resultados = {}
    for d in DOENCAS:
        res = mk.hamed_rao_modification_test(df[d].values)
        resultados[d] = {
            "trend": res.trend,
            "p": res.p,
            "tau": res.Tau,
            "slope": res.slope,
            "intercept": res.intercept,
            "significativo": res.p < 0.05,
        }
    return resultados


@st.cache_data
def calcular_decomposicao(df):
    decomp = {}
    serie_idx = df.set_index("data")
    for d in DOENCAS:
        serie = serie_idx[d]
        serie.index = pd.DatetimeIndex(serie.index).to_period("M").to_timestamp()
        stl = STL(serie, period=12, robust=True).fit()
        var_resid = np.var(stl.resid)
        var_seas_resid = np.var(stl.seasonal + stl.resid)
        fs = max(0, 1 - var_resid / var_seas_resid) if var_seas_resid > 0 else 0.0
        decomp[d] = {
            "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid,
            "fs": fs, "index": serie.index,
        }
    return decomp


@st.cache_data
def calcular_sazonalidade_kw(df):
    resultados = {}
    for d in DOENCAS:
        grupos = [df[df["mes"] == m][d].values for m in range(1, 13)]
        stat, p = stats.kruskal(*grupos)
        resultados[d] = {"H": stat, "p": p, "significativo": p < 0.05}
    return resultados


df = load_data()
mk_results = calcular_mann_kendall(df)
decomp = calcular_decomposicao(df)
kw_results = calcular_sazonalidade_kw(df)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Séries Temporais & Sazonalidade — Serra/ES")
st.caption("Mann-Kendall · Decomposição STL · Teste de Sazonalidade · Dengue, Zika, Chikungunya · 2015–2023")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FILTROS GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────
colF1, colF2 = st.columns([2, 2])

with colF1:
    doenca = st.selectbox("Doença", DOENCAS, format_func=lambda x: x.capitalize())

with colF2:
    anos_disp = sorted(df["ano"].unique())
    anos = st.slider("Período", int(min(anos_disp)), int(max(anos_disp)),
                      (int(min(anos_disp)), int(max(anos_disp))))

df_f = df[(df["ano"] >= anos[0]) & (df["ano"] <= anos[1])].copy()
cor = CORES[doenca]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total = int(df_f[doenca].sum())
media = df_f[doenca].mean()
maximo_idx = df_f[doenca].idxmax()
maximo = int(df_f.loc[maximo_idx, doenca])
mes_pico_abs = df_f.loc[maximo_idx, "data"].strftime("%b/%Y")

k1.metric(f"Total {doenca.capitalize()}", f"{total:,}".replace(",", "."),
          help=f"Soma de casos no período {anos[0]}–{anos[1]}")
k2.metric("Média mensal", f"{media:.1f}", help="Casos/mês em média no período")
k3.metric("Pico histórico", f"{maximo:,}".replace(",", "."), help=f"Mês: {mes_pico_abs}")

mk_r = mk_results[doenca]
tendencia_label = {"increasing": "Crescente ↑", "decreasing": "Decrescente ↓", "no trend": "Sem tendência"}
k4.metric("Tendência (Mann-Kendall)",
          tendencia_label.get(mk_r["trend"], mk_r["trend"]),
          help=f"p-valor={mk_r['p']:.4f}  |  Tau={mk_r['tau']:.3f}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Série temporal interativa",
    "  Teste de Mann-Kendall",
    "  Decomposição & Sazonalidade",
])

# ══════════════════════════════════════════════════════════════
# ABA 1 — SÉRIE TEMPORAL INTERATIVA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Evolução mensal — todas as doenças")
    st.markdown(
        '<div class="info-box">Série mensal de casos notificados (InfoDengue/Fiocruz — Serra/ES). '
        'Use a legenda para ocultar/exibir doenças, ou dê zoom arrastando sobre o gráfico.</div>',
        unsafe_allow_html=True
    )

    fig_all = go.Figure()
    for d in DOENCAS:
        fig_all.add_trace(go.Scatter(
            x=df_f["data"], y=df_f[d],
            name=d.capitalize(),
            line=dict(color=CORES[d], width=2),
            mode="lines",
        ))
    fig_all.update_layout(
        template="plotly_white", # Alterado para fundo claro
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        hovermode="x unified",
        yaxis_title="Casos notificados",
        height=420,
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"Total anual — {doenca.capitalize()}")
        df_ano = df_f.groupby("ano")[doenca].sum().reset_index()
        fig_bar = px.bar(
            df_ano, x="ano", y=doenca,
            color_discrete_sequence=[cor],
            template="plotly_white", # Alterado para fundo claro
            labels={"ano": "Ano", doenca: "Casos"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=0),
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Heatmap — ano × mês")
        pivot = df_f.pivot_table(index="ano", columns="mes", values=doenca, aggfunc="sum")
        pivot.columns = [MESES_LABEL[m-1] for m in pivot.columns]
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="Reds" if doenca == "dengue" else
                                   ("Greens" if doenca == "zika" else "Oranges"),
            template="plotly_white", # Alterado para fundo claro
            labels=dict(color="Casos"),
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=0),
            height=360,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 2 — TESTE DE MANN-KENDALL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Teste de Mann-Kendall (Hamed-Rao) — Detecção de Tendência")
    st.markdown("""
    <div class="info-box">
    O teste de <b>Mann-Kendall</b> verifica a existência de uma tendência monotônica
    (crescente ou decrescente) ao longo do tempo, sem assumir distribuição normal.
    A versão <b>Hamed-Rao</b> corrige a presença de autocorrelação serial, comum em
    dados epidemiológicos mensais.<br><br>
    <b>H0:</b> não há tendência &nbsp;|&nbsp; <b>H1:</b> há tendência monotônica (p &lt; 0,05 rejeita H0)
    </div>
    """, unsafe_allow_html=True)

    tabela_mk = pd.DataFrame([
        {
            "Doença": d.capitalize(),
            "Tendência": tendencia_label.get(mk_results[d]["trend"], mk_results[d]["trend"]),
            "p-valor": round(mk_results[d]["p"], 4),
            "Tau de Kendall": round(mk_results[d]["tau"], 4),
            "Sen's Slope (casos/mês)": round(mk_results[d]["slope"], 4),
            "Significativo (α=0.05)": "✅ Sim" if mk_results[d]["significativo"] else "❌ Não",
        }
        for d in DOENCAS
    ])
    st.dataframe(tabela_mk, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader(f"Série + Tendência de Sen — {doenca.capitalize()}")

    serie_full = df_f[doenca].values
    res = mk_results[doenca]
    t = np.arange(len(serie_full))
    tendencia_vals = res["slope"] * t + res["intercept"]

    fig_mk = go.Figure()
    fig_mk.add_trace(go.Scatter(
        x=df_f["data"], y=serie_full,
        name="Série observada",
        line=dict(color=cor, width=1.5),
        opacity=0.6,
    ))
    fig_mk.add_trace(go.Scatter(
        x=df_f["data"], y=tendencia_vals,
        name=f"Tendência de Sen (slope={res['slope']:.3f}/mês)",
        line=dict(color="#1f2328", width=2.5, dash="dash"), # Ajustado para linha escura
    ))
    fig_mk.update_layout(
        template="plotly_white", # Alterado para fundo claro
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=0),
        hovermode="x unified",
        yaxis_title="Casos notificados",
        height=420,
    )
    st.plotly_chart(fig_mk, use_container_width=True)

    if res["significativo"]:
        st.markdown(f"""
        <div class="result-box">
         <b>Tendência {tendencia_label.get(res['trend'], res['trend']).lower()} estatisticamente significativa</b>
        (p={res['p']:.4f} &lt; 0.05).<br>
        A inclinação estimada (Sen's slope) é de <b>{res['slope']:.3f} casos/mês</b>,
        indicando uma variação consistente ao longo de todo o período analisado.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warn-box">
         <b>Sem tendência monotônica significativa</b> (p={res['p']:.4f} ≥ 0.05).<br>
        Isso não significa ausência de variação — geralmente indica que a série é dominada
        por <b>surtos epidêmicos pontuais</b> (picos que sobem e descem), que se cancelam
        no cálculo de uma tendência linear acumulada. Veja a aba de Decomposição para
        analisar o componente de tendência local (não-linear).
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ABA 3 — DECOMPOSIÇÃO & SAZONALIDADE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Decomposição STL — {doenca.capitalize()}")
    st.markdown("""
    <div class="info-box">
    Decomposição <b>STL</b> (Seasonal-Trend decomposition using LOESS) separa a série em
    três componentes: <b>Tendência</b> (movimento de longo prazo, suavizado),
    <b>Sazonalidade</b> (padrão que se repete a cada 12 meses) e <b>Resíduo</b> (o que resta,
    incluindo surtos pontuais e ruído).
    </div>
    """, unsafe_allow_html=True)

    dec = decomp[doenca]
    comp_cols = st.columns(3)
    componentes = {"Tendência": dec["trend"], "Sazonalidade": dec["seasonal"], "Resíduo": dec["resid"]}
    cores_comp = ["#0969da", "#1a7f37", "#57606a"] # Ajustado para o tema claro

    for i, (nome, comp) in enumerate(componentes.items()):
        with comp_cols[i]:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=dec["index"], y=comp.values,
                line=dict(color=cores_comp[i], width=1.8),
                mode="lines", name=nome,
            ))
            if nome == "Resíduo":
                fig_c.add_hline(y=0, line_dash="dot", line_color="#57606a", line_width=1)
            fig_c.update_layout(
                title=dict(text=nome, font=dict(size=13)),
                template="plotly_white", # Alterado para fundo claro
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=0, l=0, r=0),
                showlegend=False,
                height=240,
            )
            st.plotly_chart(fig_c, use_container_width=True)

    st.divider()

    st.subheader("Teste de Sazonalidade — Kruskal-Wallis")
    st.markdown("""
    <div class="info-box">
    O teste de <b>Kruskal-Wallis</b> verifica se a distribuição de casos difere entre os
    12 meses do ano (equivalente não-paramétrico de uma ANOVA por postos).<br>
    <b>H0:</b> distribuição igual entre meses (sem sazonalidade) &nbsp;|&nbsp;
    <b>H1:</b> há diferença entre meses (sazonalidade)
    </div>
    """, unsafe_allow_html=True)

    tabela_kw = pd.DataFrame([
        {
            "Doença": d.capitalize(),
            "Estatística H": round(kw_results[d]["H"], 3),
            "p-valor": round(kw_results[d]["p"], 4),
            "Sazonalidade significativa": "✅ Sim" if kw_results[d]["significativo"] else "❌ Não",
            "Força Sazonal (STL, 0-1)": round(decomp[d]["fs"], 3),
        }
        for d in DOENCAS
    ])
    st.dataframe(tabela_kw, use_container_width=True, hide_index=True)

    kw_r = kw_results[doenca]
    if kw_r["significativo"]:
        st.markdown(f"""
        <div class="result-box">
         <b>Sazonalidade intra-anual estatisticamente significativa</b> para {doenca}
        (p={kw_r['p']:.4f} &lt; 0.05). A distribuição de casos varia de forma consistente
        entre os meses do ano.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warn-box">
         <b>Sem sazonalidade de calendário estatisticamente significativa</b> para {doenca}
        (p={kw_r['p']:.4f} ≥ 0.05). Os picos de casos não ocorrem de forma consistente no
        mesmo mês todos os anos — a dinâmica é melhor descrita como
        <b>epidêmica/cíclica multi-anual</b> (surtos irregulares) do que sazonal de calendário.
        Força sazonal (STL): {decomp[doenca]["fs"]:.3f} (0 = nenhuma, 1 = totalmente determinística).
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Padrão médio por mês do ano")
    media_mes = df_f.groupby("mes")[doenca].mean()
    desvio_mes = df_f.groupby("mes")[doenca].std()
    mes_pico = MESES_LABEL[media_mes.idxmax() - 1]

    fig_saz = go.Figure()
    fig_saz.add_trace(go.Bar(
        x=MESES_LABEL, y=media_mes.values,
        error_y=dict(type="data", array=desvio_mes.values, visible=True),
        marker_color=cor,
    ))
    fig_saz.update_layout(
        template="plotly_white", # Alterado para fundo claro
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=0),
        yaxis_title="Casos médios",
        height=380,
    )
    st.plotly_chart(fig_saz, use_container_width=True)
    st.caption(f" Mês de maior média histórica: **{mes_pico}** (força sazonal = {decomp[doenca]['fs']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("FAESA · Projeto Integrador III · Ciência de Dados · 2025 · Dados reais: InfoDengue/Fiocruz — Serra/ES (2015–2023)")