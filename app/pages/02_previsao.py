# -*- coding: utf-8 -*-
"""
Previsão de Arboviroses — Serra/ES
Página 02 do Dashboard · Pessoa 3 — Previsão (ML)
Modelo: SARIMA via statsmodels
"""

from io import StringIO
import urllib.request
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def hex_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


st.set_page_config(
    page_title="Previsão — Serra/ES",
    page_icon="📈",
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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
DOENCAS = ["dengue", "zika", "chikungunya"]
CORES   = {"dengue": "#f85149", "zika": "#3fb950", "chikungunya": "#d29922"}

HORIZONTES = {
    "~4 semanas  (1 mês)":  1,
    "~8 semanas  (2 meses)": 2,
    "~12 semanas (3 meses)": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# DADOS — mesma lógica do load_data() em app/main.py
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ PARA DADOS REAIS: substitua este bloco por:
#   casos = pd.read_csv("dados/casos_limpos.csv", parse_dates=["data"])
# O CSV deve ter colunas: data, dengue, zika, chikungunya

def baixar_infodengue(geocode, doenca_code, label, anos):
    import ssl
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE

    base   = "https://info.dengue.mat.br/api/alertcity"
    partes = []
    for ano in anos:
        url = (f"{base}?geocode={geocode}"
               f"&disease={doenca_code}&format=json"
               f"&ew_start=1&ew_end=53"
               f"&ey_start={ano}&ey_end={ano}")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as r:
                dados = json.loads(r.read().decode())
            if dados:
                partes.extend(dados)
        except Exception as ex:
            raise RuntimeError(f"{label} {ano}: {ex}")

    if not partes:
        raise RuntimeError(f"API retornou vazio para {label}")

    df = pd.DataFrame(partes)
    df["data"]   = pd.to_datetime(df["data_iniSE"], unit="ms")
    df["ano"]    = df["data"].dt.year
    df["mes"]    = df["data"].dt.month
    df["doenca"] = label
    df["casos"]  = df["casos"].fillna(0).astype(int)

    # Agrega semanas → meses
    mensal = (df.groupby(["ano", "mes", "doenca"])["casos"]
                .sum().reset_index())
    mensal["casos"] = mensal["casos"].astype(int)
    return mensal


@st.cache_data(ttl=86400)
def load_data():
    anos    = list(range(2015, 2024))
    geocode = 3205002  # Serra/ES

    erro_real = None
    try:
        partes = []
        for code, label in [("dengue",       "dengue"),
                             ("zika",         "zika"),
                             ("chikungunya",  "chikungunya")]:
            partes.append(baixar_infodengue(geocode, code, label, anos))

        df_long = pd.concat(partes, ignore_index=True)

        # Garante série contínua preenchendo meses sem casos com zero
        idx_completo = pd.date_range("2015-01", "2023-12", freq="MS")
        linhas = []
        for doenca in ["dengue", "zika", "chikungunya"]:
            for dt in idx_completo:
                sub = df_long[(df_long["doenca"] == doenca) &
                              (df_long["ano"]    == dt.year) &
                              (df_long["mes"]    == dt.month)]
                casos = int(sub["casos"].sum()) if len(sub) else 0
                linhas.append({"data": dt, "ano": dt.year,
                                "mes": dt.month, "doenca": doenca,
                                "casos": casos})

        df_long = pd.DataFrame(linhas)
        df_long["data"] = pd.to_datetime(df_long["data"])

        # Pivota para formato largo (uma coluna por doença)
        df_wide = df_long.pivot_table(
            index="data", columns="doenca",
            values="casos", aggfunc="sum"
        ).reset_index()
        df_wide.columns.name = None
        return df_wide[["data", "dengue", "zika", "chikungunya"]]

    except Exception as e:
        erro_real = str(e)

    # Fallback simulado — mostra o erro real para diagnóstico
    st.warning(f"⚠️ API InfoDengue indisponível: **{erro_real}**. Usando dados simulados.")
    np.random.seed(42)
    datas = pd.date_range("2015-01-01", "2023-12-01", freq="MS")
    n     = len(datas)
    t     = np.arange(n)
    sazon = np.sin(2 * np.pi * (t % 12) / 12 - np.pi / 2) * 0.5 + 0.5
    return pd.DataFrame({
        "data":        datas,
        "dengue":      (sazon * 400 + np.random.randint(50, 150, n)).astype(int),
        "zika":        (sazon * 100 + np.random.randint(10,  40, n)).astype(int),
        "chikungunya": (sazon * 60  + np.random.randint(5,   25, n)).astype(int),
    })


@st.cache_data
def treinar_modelos():
    casos = load_data()
    casos["data"] = pd.to_datetime(casos["data"])

    resultados = {}
    TESTE_N = 12

    for doenca in DOENCAS:
        serie = casos.set_index("data")[doenca]
        serie.index = pd.DatetimeIndex(serie.index).to_period("M").to_timestamp()
        treino, teste = serie.iloc[:-TESTE_N], serie.iloc[-TESTE_N:]

        fit_eval = SARIMAX(
            treino,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        pred_eval = fit_eval.get_forecast(steps=TESTE_N).predicted_mean
        mae  = round(mean_absolute_error(teste, pred_eval), 1)
        rmse = round(float(np.sqrt(mean_squared_error(teste, pred_eval))), 1)

        fit_full = SARIMAX(
            serie,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        prev_3m  = fit_full.get_forecast(steps=3)
        pred_fut = prev_3m.predicted_mean
        ci_fut   = prev_3m.conf_int(alpha=0.20)

        resultados[doenca] = {
            "serie":     serie,
            "treino":    treino,
            "teste":     teste,
            "pred_eval": pred_eval,
            "pred_fut":  pred_fut,
            "ci_fut":    ci_fut,
            "mae":       mae,
            "rmse":      rmse,
        }

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO + TREINO
# ─────────────────────────────────────────────────────────────────────────────
casos = load_data()

with st.spinner("🔧 Ajustando modelos SARIMA... (pode levar alguns segundos)"):
    resultados = treinar_modelos()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Previsão de Arboviroses — Serra/ES")
st.caption("Modelo SARIMA · Dengue · Zika · Chikungunya · 2015–2023")
st.divider()

st.markdown("""
<div class="info-box">
    Modelo <strong>SARIMA(1,1,1)(1,1,1,12)</strong> treinado nos dados mensais de 2015–2022
    e avaliado no ano de 2023. Previsões futuras geradas re-treinando com toda a série.
    <br>⚠️ <em>Dados simulados para fins acadêmicos.</em>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────────────────────
colF1, colF2 = st.columns([2, 2])

with colF1:
    doenca = st.selectbox(
        "Doença", DOENCAS,
        format_func=lambda x: x.capitalize()
    )

with colF2:
    horizonte_label = st.selectbox("Horizonte de previsão", list(HORIZONTES.keys()))

horizonte_meses = HORIZONTES[horizonte_label]
r = resultados[doenca]
cor = CORES[doenca]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

k1.metric("MAE",  f"{r['mae']} casos/mês",   help="Erro absoluto médio (teste 2023)")
k2.metric("RMSE", f"{r['rmse']} casos/mês",  help="Raiz do erro quadrático médio")

val_h = r["pred_fut"].iloc[horizonte_meses - 1]
lb_h  = r["ci_fut"].iloc[horizonte_meses - 1, 0]
ub_h  = r["ci_fut"].iloc[horizonte_meses - 1, 1]

k3.metric(
    f"Previsão {horizonte_label.split('(')[0].strip()}",
    f"{val_h:.0f} casos",
    help=f"Previsão pontual para {r['pred_fut'].index[horizonte_meses-1].strftime('%b/%Y')}"
)
k4.metric(
    "IC 80%",
    f"{lb_h:.0f} – {ub_h:.0f}",
    help="Intervalo de confiança de 80% para o horizonte selecionado"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ABAS: Avaliação | Previsão futura
# ─────────────────────────────────────────────────────────────────────────────
tab_aval, tab_prev = st.tabs(["🔍  Avaliação — Previsão vs Real (2023)", "🔮  Previsão futura"])

# ── ABA 1: Previsão vs Real ───────────────────────────────────────────────────
with tab_aval:
    st.subheader(f"Previsão vs Real — {doenca.capitalize()} (2023)")

    fig_eval = go.Figure()

    # Histórico (treino)
    fig_eval.add_trace(go.Scatter(
        x=r["treino"].index, y=r["treino"].values,
        name="Histórico (treino)",
        line=dict(color=cor, width=1.5),
        opacity=0.4,
    ))

    # Real (teste)
    fig_eval.add_trace(go.Scatter(
        x=r["teste"].index, y=r["teste"].values,
        name="Real (2023)",
        line=dict(color="white", width=2.5),
    ))

    # Previsão
    fig_eval.add_trace(go.Scatter(
        x=r["pred_eval"].index, y=r["pred_eval"].values,
        name="Previsão SARIMA",
        line=dict(color=cor, width=2.5, dash="dash"),
    ))

   # 1. Converte a data do Pandas para milissegundos (float/int)
    posicao_x_ms = r["teste"].index[0].value // 10**6

    # 2. Adiciona a linha usando o valor numérico
    fig_eval.add_vline(
        x=posicao_x_ms,
        line_dash="dot", line_color="#8b949e", line_width=1,
        annotation_text="início do teste",
        annotation_font_color="#8b949e",
        annotation_font_size=11,
    )
    
    # 3. Força o tipo do eixo X a continuar sendo tratado como data
    fig_eval.update_xaxes(type="date")

    fig_eval.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=30, b=0),
        yaxis_title="Casos notificados",
        height=420,
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Tabela de métricas comparativas
    st.subheader("Métricas por doença")
    df_metricas = pd.DataFrame([
        {
            "Doença":       d.capitalize(),
            "MAE (casos/mês)":  resultados[d]["mae"],
            "RMSE (casos/mês)": resultados[d]["rmse"],
        }
        for d in DOENCAS
    ])
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

# ── ABA 2: Previsão futura ────────────────────────────────────────────────────
with tab_prev:
    st.subheader(f"Previsão futura — {doenca.capitalize()} · {horizonte_label}")

    # Últimos 24 meses de histórico + previsão
    historico_rec = r["serie"].iloc[-24:]
    pred_plot     = r["pred_fut"].iloc[:horizonte_meses]
    ci_plot       = r["ci_fut"].iloc[:horizonte_meses]

    fig_fut = go.Figure()

    # Histórico recente
    fig_fut.add_trace(go.Scatter(
        x=historico_rec.index, y=historico_rec.values,
        name="Histórico (últimos 24 meses)",
        line=dict(color=cor, width=2),
        opacity=0.6,
    ))

    # IC
    fig_fut.add_trace(go.Scatter(
        x=list(ci_plot.index) + list(ci_plot.index[::-1]),
        y=list(ci_plot.iloc[:, 0]) + list(ci_plot.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor=hex_rgba(cor, 0.2),
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 80%",
    ))

    # Previsão pontual
    fig_fut.add_trace(go.Scatter(
        x=pred_plot.index, y=pred_plot.values,
        name=f"Previsão ({horizonte_label})",
        mode="lines+markers",
        line=dict(color=cor, width=3, dash="dot"),
        marker=dict(size=10, symbol="circle"),
    ))

    # Anotações nos pontos de previsão
    semanas_labels = ["~4 sem", "~8 sem", "~12 sem"]
    for j in range(horizonte_meses):
        fig_fut.add_annotation(
            x=pred_plot.index[j],
            y=pred_plot.values[j],
            text=f"<b>{pred_plot.values[j]:.0f}</b><br>{semanas_labels[j]}",
            showarrow=True, arrowhead=2,
            ax=0, ay=-40,
            font=dict(size=11, color=cor),
            bgcolor="#161b22",
            bordercolor=cor,
        )

    fig_fut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=30, b=0),
        yaxis_title="Casos notificados",
        height=440,
    )
    st.plotly_chart(fig_fut, use_container_width=True)

    # Tabela de previsão
    df_prev = pd.DataFrame({
        "Mês":       [d.strftime("%b/%Y") for d in r["pred_fut"].index[:horizonte_meses]],
        "Horizonte": semanas_labels[:horizonte_meses],
        "Previsão":  r["pred_fut"].values[:horizonte_meses].round(0).astype(int),
        "IC inf.":   r["ci_fut"].iloc[:horizonte_meses, 0].round(0).astype(int).values,
        "IC sup.":   r["ci_fut"].iloc[:horizonte_meses, 1].round(0).astype(int).values,
    })
    st.dataframe(df_prev, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("FAESA · Projeto Integrador III · Ciência de Dados · 2025 · dados simulados para fins acadêmicos")
