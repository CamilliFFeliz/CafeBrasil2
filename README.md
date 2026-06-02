# Café Brasil - Análise de Dados

Projeto de análise de dados sobre café no Brasil, com back-end em Python para tratamento, estatística, regressão e previsão, além de uma interface web estática pronta para GitHub Pages.

## Estrutura principal

```text
CafeBrasil2-main/
├── index.html
├── style.css
├── script.js
├── analysis_results.json
├── requirements.txt
├── relatorio_regressao.txt
├── assets/
│   └── info.jpeg
└── Projeto_do_Cafe_no_Brasil/
    ├── Area_Colhida.xlsx
    ├── Area_Plantada.xlsx
    ├── analise_regressao_dashboard.py
    └── Readme
```

## Instalação do back-end

```bash
pip install -r requirements.txt
```

## Gerar dados para a interface web

```bash
python Projeto_do_Cafe_no_Brasil/analise_regressao_dashboard.py --export-static
```

Esse comando atualiza o arquivo `analysis_results.json`, usado diretamente pelo `index.html`.

## Gerar relatório e gráficos

```bash
python Projeto_do_Cafe_no_Brasil/analise_regressao_dashboard.py --report
```

O comando cria ou atualiza:

```text
analysis_results.json
relatorio_regressao.txt
graficos/
```

## Executar dashboard Streamlit opcional

```bash
streamlit run Projeto_do_Cafe_no_Brasil/analise_regressao_dashboard.py
```

## Publicação no GitHub Pages

A interface estática já está na raiz do projeto. Para publicar, envie o repositório para o GitHub e habilite o GitHub Pages apontando para a branch principal e a pasta raiz. Os arquivos necessários são `index.html`, `style.css`, `script.js`, `analysis_results.json` e `assets/info.jpeg`.

## Melhorias aplicadas

O pipeline mantém separadas as bases de Área Colhida e Área Plantada, preservando a origem dos registros e evitando perda de dados por deduplicação incorreta. Também foram adicionadas validações de colunas obrigatórias, limpeza monetária mais robusta, métricas de qualidade, regressões, previsão mensal e exportação JSON para consumo no front-end.
