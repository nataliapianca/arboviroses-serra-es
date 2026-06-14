from pathlib import Path
import json
import urllib.request

import geopandas as gpd
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


CODIGO_MUN_IBGE_SERRA = "3205002"
GEOCODE_SERRA = 3205002
FONTE_MALHA = "IBGE Censo 2022"
FONTE_CASOS = "InfoDengue/Fiocruz - Serra/ES"
DOENCAS = ["dengue", "zika", "chikungunya"]
NOMES_DOENCAS = {"dengue": "Dengue", "zika": "Zika", "chikungunya": "Chikungunya"}
CORES_DOENCAS = {"dengue": "rgb(248,81,73)", "zika": "rgb(63,185,80)", "chikungunya": "rgb(210,153,34)"}
COLUNAS_TABELA = ["CD_MUN", "NM_MUN", "CD_BAIRRO", "NM_BAIRRO", "NM_DIST"]
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_MALHA = RAIZ_PROJETO / "data" / "geografia" / "bairros_ibge" / "ES_bairros_CD2022.shp"
CAMINHO_CASOS_MENSAIS = RAIZ_PROJETO / "data" / "comparativo" / "comparativo_mensal.csv"


@st.cache_data
def carregar_malha_bairros():
    if not CAMINHO_MALHA.exists():
        raise FileNotFoundError(f"Malha de bairros ausente em {CAMINHO_MALHA.relative_to(RAIZ_PROJETO).as_posix()}")

    gdf = gpd.read_file(CAMINHO_MALHA)
    colunas_faltantes = [coluna for coluna in [*COLUNAS_TABELA, "geometry"] if coluna not in gdf.columns]
    if colunas_faltantes:
        raise KeyError(f"Colunas ausentes na malha de bairros: {colunas_faltantes}")

    gdf["CD_MUN"] = gdf["CD_MUN"].astype(str)
    gdf_serra = gdf[gdf["CD_MUN"].eq(CODIGO_MUN_IBGE_SERRA)].copy()
    if gdf_serra.empty:
        raise ValueError(f"O filtro CD_MUN == {CODIGO_MUN_IBGE_SERRA} retornou zero bairros.")

    crs_original = str(gdf_serra.crs)
    if gdf_serra.crs is not None and gdf_serra.crs.to_epsg() != 4326:
        gdf_serra = gdf_serra.to_crs(epsg=4326)

    tabela = (
        gdf_serra[COLUNAS_TABELA]
        .drop_duplicates("CD_BAIRRO")
        .sort_values("NM_BAIRRO")
        .reset_index(drop=True)
    )
    return gdf_serra, tabela, crs_original


def baixar_infodengue(doenca, ano):
    url = (
        "https://info.dengue.mat.br/api/alertcity"
        f"?geocode={GEOCODE_SERRA}"
        f"&disease={doenca}"
        "&format=json"
        "&ew_start=1"
        "&ew_end=53"
        f"&ey_start={ano}"
        f"&ey_end={ano}"
    )
    requisicao = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(requisicao, timeout=30) as resposta:
        dados = json.loads(resposta.read().decode("utf-8"))
    return pd.DataFrame(dados)


@st.cache_data(ttl=86400)
def carregar_casos_municipais():
    if CAMINHO_CASOS_MENSAIS.exists():
        dados = pd.read_csv(CAMINHO_CASOS_MENSAIS, parse_dates=["data"])
        dados_longos = dados.melt(id_vars="data", value_vars=DOENCAS, var_name="doenca", value_name="casos")
        dados_longos["ano"] = dados_longos["data"].dt.year
        dados_longos["mes"] = dados_longos["data"].dt.month
        dados_longos["casos"] = dados_longos["casos"].fillna(0).astype(int)
        return dados_longos, FONTE_CASOS, CAMINHO_CASOS_MENSAIS.relative_to(RAIZ_PROJETO).as_posix()

    partes = []
    for doenca in DOENCAS:
        for ano in range(2015, 2024):
            dados_ano = baixar_infodengue(doenca, ano)
            if len(dados_ano):
                dados_ano["data"] = pd.to_datetime(dados_ano["data_iniSE"], unit="ms")
                dados_ano["ano"] = dados_ano["data"].dt.year
                dados_ano["mes"] = dados_ano["data"].dt.month
                dados_ano["doenca"] = doenca
                partes.append(dados_ano[["data", "ano", "mes", "doenca", "casos"]])

    if not partes:
        raise RuntimeError("Falha ao carregar dados do InfoDengue para Serra/ES.")

    dados_longos = pd.concat(partes, ignore_index=True)
    dados_longos = (
        dados_longos
        .groupby(["data", "ano", "mes", "doenca"], as_index=False)["casos"]
        .sum()
    )
    dados_longos["casos"] = dados_longos["casos"].fillna(0).astype(int)
    return dados_longos, FONTE_CASOS, "API InfoDengue"


def formatar_inteiro(valor):
    return f"{int(round(valor)):,}".replace(",", ".")


def formatar_decimal(valor):
    texto = f"{float(valor):,.1f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def preparar_geojson(gdf):
    dados = gdf[[*COLUNAS_TABELA, "geometry"]].copy()
    dados["tooltip"] = (
        "<b>" + dados["NM_BAIRRO"].astype(str) + "</b><br>"
        "Código do bairro: " + dados["CD_BAIRRO"].astype(str) + "<br>"
        "Distrito: " + dados["NM_DIST"].astype(str) + "<br>"
        "Município: " + dados["NM_MUN"].astype(str)
    )
    return json.loads(dados.to_json())


def criar_mapa(gdf):
    geojson = preparar_geojson(gdf)
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    visualizacao = pdk.ViewState(
        latitude=(min_lat + max_lat) / 2,
        longitude=(min_lon + max_lon) / 2,
        zoom=10.8,
        pitch=0,
    )
    camada = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color=[88, 166, 255, 70],
        get_line_color=[230, 237, 243, 190],
        get_line_width=80,
        line_width_min_pixels=1,
    )
    return pdk.Deck(
        layers=[camada],
        initial_view_state=visualizacao,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": "{tooltip}",
            "style": {
                "backgroundColor": "rgb(22, 27, 34)",
                "color": "rgb(230, 237, 243)",
                "fontFamily": "IBM Plex Sans, sans-serif",
                "fontSize": "13px",
            },
        },
    )


def aplicar_estilo():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: rgb(13, 17, 23); color: rgb(230, 237, 243); }
h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: rgb(230, 237, 243) !important;
    letter-spacing: 0;
}
[data-testid="metric-container"] {
    background: rgb(22, 27, 34);
    border: 1px solid rgb(33, 38, 45);
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: rgb(139, 148, 158) !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important;
    color: rgb(230, 237, 243) !important;
}
hr { border-color: rgb(33, 38, 45) !important; }
.info-box {
    background: rgb(22, 27, 34);
    border: 1px solid rgb(33, 38, 45);
    border-left: 3px solid rgb(88, 166, 255);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: rgb(139, 148, 158);
    margin-bottom: 1rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def filtrar_casos(dados, doencas, anos):
    return dados[dados["doenca"].isin(doencas) & dados["ano"].between(anos[0], anos[1])].copy()


def preparar_resumo_doencas(casos, total_bairros):
    resumo = (
        casos
        .groupby("doenca", as_index=False)["casos"]
        .sum()
        .sort_values("casos", ascending=False)
        .reset_index(drop=True)
    )
    total = resumo["casos"].sum()
    resumo["participacao_percentual"] = resumo["casos"].div(total).mul(100) if total else 0
    resumo["media_municipal_por_bairro"] = resumo["casos"].div(total_bairros)
    resumo["Doença"] = resumo["doenca"].map(NOMES_DOENCAS)
    return resumo


def preparar_periodos_criticos(casos, total_bairros):
    casos_anuais = casos.groupby("ano", as_index=False)["casos"].sum().sort_values("casos", ascending=False)
    casos_anuais["media_municipal_por_bairro"] = casos_anuais["casos"].div(total_bairros)
    casos_mensais = casos.groupby("data", as_index=False)["casos"].sum().sort_values("casos", ascending=False)
    casos_mensais["media_municipal_por_bairro"] = casos_mensais["casos"].div(total_bairros)
    picos_doenca = (
        casos
        .sort_values("casos", ascending=False)
        .groupby("doenca", as_index=False)
        .first()[["doenca", "data", "casos"]]
    )
    picos_doenca["Doença"] = picos_doenca["doenca"].map(NOMES_DOENCAS)
    picos_doenca["Mês de pico"] = picos_doenca["data"].dt.strftime("%m/%Y")
    picos_doenca["Média municipal por bairro"] = picos_doenca["casos"].div(total_bairros)
    picos_doenca = picos_doenca[["Doença", "Mês de pico", "casos", "Média municipal por bairro"]].rename(columns={"casos": "Casos"})
    return casos_anuais.head(5), casos_mensais.head(5), picos_doenca


def main():
    st.set_page_config(page_title="Mapa dos Bairros e Contexto Epidemiológico Municipal", layout="wide", initial_sidebar_state="collapsed")
    aplicar_estilo()

    st.title("Mapa dos Bairros e Contexto Epidemiológico Municipal")
    st.caption("Serra/ES · malha territorial dos bairros · série municipal de arboviroses")

    try:
        gdf_bairros_serra, tabela_bairros, crs_original = carregar_malha_bairros()
        casos_municipais, fonte_casos, origem_casos = carregar_casos_municipais()
    except Exception as erro:
        st.error(f"Falha no carregamento dos dados necessários: {erro}")
        st.stop()

    anos_disponiveis = sorted(casos_municipais["ano"].dropna().astype(int).unique())
    opcoes_doencas = st.multiselect(
        "Doenças",
        DOENCAS,
        default=DOENCAS,
        format_func=lambda valor: NOMES_DOENCAS[valor],
    )
    intervalo_anos = st.slider("Período", min_value=min(anos_disponiveis), max_value=max(anos_disponiveis), value=(min(anos_disponiveis), max(anos_disponiveis)))

    if not opcoes_doencas:
        st.warning("Selecionem pelo menos uma doença para visualizar os indicadores municipais.")
        st.stop()

    total_bairros = len(tabela_bairros)
    casos_filtrados = filtrar_casos(casos_municipais, opcoes_doencas, intervalo_anos)
    resumo_doencas = preparar_resumo_doencas(casos_filtrados, total_bairros)
    anos_criticos, meses_criticos, picos_doenca = preparar_periodos_criticos(casos_filtrados, total_bairros)
    total_casos = int(casos_filtrados["casos"].sum())
    media_municipal_por_bairro = total_casos / total_bairros
    maior_doenca = NOMES_DOENCAS[resumo_doencas.iloc[0]["doenca"]] if len(resumo_doencas) else "Sem seleção"
    periodo_texto = f"{intervalo_anos[0]} a {intervalo_anos[1]}"

    coluna_1, coluna_2, coluna_3, coluna_4, coluna_5 = st.columns(5)
    coluna_1.metric("Casos municipais", formatar_inteiro(total_casos))
    coluna_2.metric("Maior total", maior_doenca)
    coluna_3.metric("Período", periodo_texto)
    coluna_4.metric("Bairros", formatar_inteiro(total_bairros))
    coluna_5.metric("Média mun./bairro", formatar_decimal(media_municipal_por_bairro))

    st.markdown(
        f"""
<div class="info-box">
Integramos a malha oficial dos bairros ao histórico municipal de arboviroses. A média municipal por bairro é um indicador derivado do total municipal dividido pelos {total_bairros} bairros reconhecidos.
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Mapa territorial dos bairros")
    st.pydeck_chart(criar_mapa(gdf_bairros_serra), use_container_width=True, height=520)

    st.subheader("Resumo municipal por doença")
    resumo_exibicao = resumo_doencas.copy()
    tabela_resumo = resumo_exibicao[["Doença", "casos", "participacao_percentual", "media_municipal_por_bairro"]].rename(
        columns={
            "casos": "Casos",
            "participacao_percentual": "Participação (%)",
            "media_municipal_por_bairro": "Média municipal por bairro",
        }
    )
    st.dataframe(tabela_resumo, hide_index=True, use_container_width=True)

    coluna_media, coluna_comparativo = st.columns(2)
    with coluna_media:
        fig_media = px.bar(
            resumo_exibicao,
            x="Doença",
            y="media_municipal_por_bairro",
            color="doenca",
            color_discrete_map=CORES_DOENCAS,
            labels={"media_municipal_por_bairro": "Média municipal por bairro", "Doença": "Doença"},
        )
        fig_media.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(fig_media, use_container_width=True)
    with coluna_comparativo:
        fig_comparativo = px.bar(
            resumo_exibicao,
            x="Doença",
            y="casos",
            color="doenca",
            color_discrete_map=CORES_DOENCAS,
            labels={"casos": "Casos", "Doença": "Doença"},
        )
        fig_comparativo.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(fig_comparativo, use_container_width=True)

    st.subheader("Evolução municipal das arboviroses")
    fig_evolucao = px.line(
        casos_filtrados.sort_values("data"),
        x="data",
        y="casos",
        color="doenca",
        color_discrete_map=CORES_DOENCAS,
        labels={"data": "Mês", "casos": "Casos", "doenca": "Doença"},
        markers=False,
    )
    fig_evolucao.for_each_trace(lambda trace: trace.update(name=NOMES_DOENCAS.get(trace.name, trace.name)))
    fig_evolucao.update_layout(template="plotly_dark", height=420, legend_title_text="Doença", hovermode="x unified")
    st.plotly_chart(fig_evolucao, use_container_width=True)

    st.subheader("Períodos críticos municipais")
    coluna_critica_1, coluna_critica_2, coluna_critica_3 = st.columns(3)
    with coluna_critica_1:
        st.markdown("**Anos com maior volume**")
        anos_exibicao = anos_criticos.rename(columns={"ano": "Ano", "casos": "Casos", "media_municipal_por_bairro": "Média municipal por bairro"})
        st.dataframe(anos_exibicao, hide_index=True, use_container_width=True)
    with coluna_critica_2:
        st.markdown("**Meses com maior volume**")
        meses_exibicao = meses_criticos.copy()
        meses_exibicao["Mês"] = meses_exibicao["data"].dt.strftime("%m/%Y")
        meses_exibicao = meses_exibicao[["Mês", "casos", "media_municipal_por_bairro"]].rename(columns={"casos": "Casos", "media_municipal_por_bairro": "Média municipal por bairro"})
        st.dataframe(meses_exibicao, hide_index=True, use_container_width=True)
    with coluna_critica_3:
        st.markdown("**Pico mensal por doença**")
        st.dataframe(picos_doenca, hide_index=True, use_container_width=True)

    fig_anos = px.bar(
        anos_criticos.sort_values("ano"),
        x="ano",
        y="casos",
        labels={"ano": "Ano", "casos": "Casos"},
    )
    fig_anos.update_layout(template="plotly_dark", height=320, showlegend=False)
    st.plotly_chart(fig_anos, use_container_width=True)

    st.subheader("Bairros reconhecidos")
    tabela_exibicao = tabela_bairros.rename(
        columns={
            "CD_MUN": "Código do município",
            "NM_MUN": "Município",
            "CD_BAIRRO": "Código do bairro",
            "NM_BAIRRO": "Bairro",
            "NM_DIST": "Distrito",
        }
    )
    st.dataframe(tabela_exibicao, hide_index=True, use_container_width=True)

    st.subheader("Série municipal consolidada")
    tabela_municipal = (
        casos_filtrados
        .groupby(["ano", "doenca"], as_index=False)["casos"]
        .sum()
        .sort_values(["ano", "doenca"])
    )
    tabela_municipal["Doença"] = tabela_municipal["doenca"].map(NOMES_DOENCAS)
    tabela_municipal["Média municipal por bairro"] = tabela_municipal["casos"].div(total_bairros)
    tabela_municipal = tabela_municipal[["ano", "Doença", "casos", "Média municipal por bairro"]].rename(columns={"ano": "Ano", "casos": "Casos"})
    st.dataframe(tabela_municipal, hide_index=True, use_container_width=True)

    with st.expander("Detalhes técnicos"):
        st.write(f"Fonte da malha: `{FONTE_MALHA}`")
        st.write(f"Fonte dos casos: `{fonte_casos}`")
        st.write(f"Origem carregada na página: `{origem_casos}`")
        st.write(f"Caminho da malha: `{CAMINHO_MALHA.relative_to(RAIZ_PROJETO).as_posix()}`")
        st.write(f"CRS original: `{crs_original}`")
        st.write(f"CRS em uso no mapa: `{gdf_bairros_serra.crs}`")


if __name__ == "__main__":
    main()
