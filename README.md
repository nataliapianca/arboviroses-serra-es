# 🦟 Arboviroses Serra/ES — Análise de Doenças Tropicais

Projeto desenvolvido para a disciplina **Projeto Integrador III – Ciência de Dados** da FAESA, curso de Ciência da Computação.
**Orientador:** Prof. Howard Cruz

---

## 📌 Sobre o Projeto

Este projeto analisa a **distribuição geográfica, sazonalidade e projeções de arboviroses** — Dengue, Zika e Chikungunya — no município de **Serra/ES**, abrangendo o período de 2015 a 2023. 

Utilizando técnicas avançadas de Ciência de Dados e Estatística, integramos a malha geográfica municipal (shapefiles do IBGE Censo 2022) com o histórico de notificações reais (SINAN/DATASUS via InfoDengue) para gerar painéis, relatórios e KPIs que apoiem a tomada de decisão da **vigilância epidemiológica local**.

---

## 🎯 Problema e Impacto Social

Serra é historicamente um dos municípios com os maiores índices de notificações de arboviroses no Espírito Santo. Este projeto visa preencher a lacuna de análises interativas integradas que facilitem a visualização espacial de bairros, a detecção estatística de tendências de longo prazo e a modelagem preditiva de surtos. 

**Sociedade Impactada:**
* **Gestores de Saúde Coletiva:** Auxiliando no direcionamento de campanhas de combate aos vetores (*Aedes aegypti*) nos bairros críticos.
* **População de Serra/ES:** Oferecendo transparência sobre o histórico de contágios e previsões para planejamento individual.

---

## 🧪 Metodologias e Testes Estatísticos Aplicados

Para garantir o rigor científico exigido na Ciência da Computação, foram aplicadas as seguintes modelagens nas séries mensais (2015–2023):

1. **Teste de Mann-Kendall (Modificação de Hamed-Rao):**
   * **Objetivo:** Detectar a presença de tendências monotônicas (crescentes ou decrescentes) ao longo do tempo.
   * **Diferencial:** A correção de Hamed-Rao trata a autocorrelação serial, comum em dados mensais de saúde, evitando falsos positivos.

2. **Decomposição STL (Seasonal-Trend decomposition using LOESS):**
   * **Objetivo:** Decompor a série histórica em três componentes independentes: **Tendência** (longo prazo), **Sazonalidade** (ciclo anual de 12 meses) e **Resíduos** (flutuações de surtos atípicos e ruído).

3. **Teste de Kruskal-Wallis:**
   * **Objetivo:** Verificar estatisticamente se a distribuição de casos varia significativamente entre os 12 meses do ano (sazonalidade de calendário).

4. **Modelagem Preditiva SARIMA (Seasonal ARIMA):**
   * **Modelo:** $SARIMA(1,1,1)(1,1,1)_{12}$ ajustado via `statsmodels`.
   * **Objetivo:** Projetar o volume de casos para horizontes de 1 a 3 meses (aproximadamente 4, 8 e 12 semanas) com intervalos de confiança de 80%.
   * **Avaliação:** Modelo avaliado no conjunto de teste (ano de 2023) utilizando as métricas **MAE** (Erro Absoluto Médio) e **RMSE** (Raiz do Erro Quadrático Médio).

---

## 🗂️ Estrutura do Repositório

```text
├── app/
│   ├── pages/
│   │   ├── 01_series_temporais.py  # Análise STL, Mann-Kendall e Kruskal-Wallis
│   │   ├── 02_previsao.py          # Projeções SARIMA (previsto vs real e futuro)
│   │   ├── 03_mapa_bairros.py      # Malha espacial dos bairros (Pydeck/GeoPandas)
│   │   └── 04_comparativo.py       # Comparativos gerais e tabela para download
│   └── main.py                     # Arquivo principal do Streamlit (Navegação)
├── data/
│   ├── comparativo/
│   │   ├── comparativo_anual.csv   # Histórico anual consolidado
│   │   ├── comparativo_mensal.csv  # Histórico mensal (2015–2023) (Real)
│   │   └── resumo_comparativo.csv  # Métricas consolidadas das três doenças
│   └── geografia/
│       └── bairros_ibge/           # Shapefiles do Censo IBGE 2022 para Serra/ES
├── notebooks/                      # Notebooks Jupyter usados no desenvolvimento da EDA
├── serie-temporal/                 # Documentos de modelagem e arquivos auxiliares
├── requirements.txt                # Lista de dependências do Python
└── README.md                       # Documento de apresentação
```

---

## ⚙️ Instalação e Execução Local

### Pré-requisitos
* Python 3.9 ou superior instalado.

### Passo 1: Clonar o repositório
```bash
git clone https://github.com/seu-usuario/arboviroses-serra-es.git
cd arboviroses-serra-es
```

### Passo 2: Criar e ativar um ambiente virtual (Recomendado)
```bash
# No Windows:
python -m venv venv
venv\Scripts\activate

# No macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instalar as dependências
```bash
pip install -r requirements.txt
```

### Passo 4: Executar o Dashboard Streamlit
```bash
streamlit run app/main.py
```
O aplicativo abrirá automaticamente no seu navegador no endereço `http://localhost:8501`.

---

## 📈 Resumo das Fontes de Dados

| Fonte | Descrição |
|-------|-----------|
| **DATASUS / SINAN (via InfoDengue)** | Notificações mensais de Dengue, Zika e Chikungunya (2015–2023) |
| **IBGE (Censo 2022)** | Dados geográficos da malha setorial de bairros do município de Serra/ES |
| **INMET** | Temperatura e precipitação mensais integrados às análises de correlação |
