import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # Importação para o modelo de previsão

# --- CONFIGURAÇÃO GLOBAL ---
# Formata a exibição de números de ponto flutuante para duas casas decimais
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# --- ETAPA 1: CARREGAMENTO DOS DADOS ---
def carregar_dados(caminho_arquivo):
    """
    Carrega os dados de um arquivo Excel e trata possíveis erros.
    """
    try:
        df = pd.read_excel(caminho_arquivo, sheet_name=0)
        print(f"---- Dados Brutos de '{caminho_arquivo}' Carregados com Sucesso ----")
        print(df.head())
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
        return df
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None

# --- ETAPA 2: LIMPEZA E PREPARAÇÃO DOS DADOS ---
def limpar_dados_monetarios(df):
    """
    Limpa e converte as colunas de moeda ('Real', 'Dolar') para o formato numérico.
    """
    print("\n--- Limpando dados das colunas de moeda ---")
    
    def limpar_moeda(valor):
        if isinstance(valor, (int, float)):
            return valor
        s = str(valor).strip().lower()
        if s == 'nan':
            return np.nan
        s = s.replace('r$', '').replace('.', '').replace(',', '.').strip()
        return pd.to_numeric(s, errors='coerce')

    for col in ["Real", "Dolar"]:
        if col in df.columns:
            df[col] = df[col].apply(limpar_moeda)
            
    print("\n---- Dados Limpos com Sucesso ----")
    print(df.head())
    return df

# --- ETAPA 3: ANÁLISE ESTATÍSTICA ---
def calcular_estatisticas(df, nome_analise):
    """
    Seleciona colunas e calcula as estatísticas descritivas para um DataFrame.
    """
    print(f"\n--- Calculando Estatísticas para: {nome_analise} ---")
    
    dados_analise = df[["Cafe_Arabica", "Cafe_Canephora", "Real", "Dolar"]].copy()

    # Média - É a soma de todos os valores dividida pelo número total de valores.
    print("\n____Média:____")
    print(dados_analise.mean())

    # Moda - É o valor que ocorre com mais frequência em um conjunto de dados.
    print("\n____Moda:____")
    print(dados_analise.mode().iloc[0])

    # ... (outros prints de estatísticas que você já tinha) ...
            
    # Correlação - Indica a força e a direção da relação linear entre duas variáveis.
    correlacao = dados_analise.corr()
    print("\n____Correlação:____")
    print(correlacao)
    
    return dados_analise, correlacao

# --- ETAPA 4: GERAÇÃO DE GRÁFICOS DESCRITIVOS ---
def gerar_graficos_descritivos(dados, correlacao, nome_analise):
    """
    Gera os gráficos de análise descritiva para um DataFrame.
    """
    print(f"\n--- Gerando Gráficos Descritivos para: {nome_analise} ---")

    # Gráfico 1: Histograma
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5)) 
    fig.suptitle(f"Distribuição das Variáveis - {nome_analise}", fontsize=16)
    for i, coluna in enumerate(dados.columns):
        ax = axes[i]
        serie = dados[coluna].dropna()
        sns.histplot(serie, bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax)
        ax.axvline(serie.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {serie.mean():.2f}')
        ax.axvline(serie.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {serie.median():.2f}')
        ax.set_title(coluna.replace("_", " "))
        ax.set_xlabel("Valores")
        ax.set_ylabel("Frequência")
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Gráfico 2: Boxplot e Gráfico 3: Mapa de Calor
    # (Mantidos como no seu código original)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dados, palette="Set2")
    plt.title(f"Boxplot - Dispersão dos Dados - {nome_analise}")
    plt.ylabel("Valores")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Mapa de Correlação entre Variáveis - {nome_analise}")
    plt.show()

# --- ETAPA 5: ANÁLISE COMPARATIVA DE PRODUÇÃO ---
def analisar_producao(df_plantada, df_colhida):
    """
    Compara os dados de área plantada e colhida para analisar a eficiência.
    """
    print("\nINICIANDO ANÁLISE COMPARATIVA DE PRODUÇÃO\n")


    # (Código da análise comparativa que já criamos)
    # ...
    
# --- ETAPA 6: PREVISÃO DE SÉRIES TEMPORAIS COM SARIMA ---
def prever_precos_cafe(df_plantada, df_colhida):
    """
    Unifica os dados, treina um modelo de série temporal (SARIMA) e prevê
    os preços futuros do café (coluna 'Real').
    """
    print("\nINICIANDO ANÁLISE PREDITIVA (SÉRIE TEMPORAL)\n")

    # Filtra os dados para analisar apenas o nível 'Brasil'
    plantio_br = df_plantada[df_plantada['Localidade'] == 'Brasil'].copy()
    colheita_br = df_colhida[df_colhida['Localidade'] == 'Brasil'].copy()

    # Define a 'Data' como o índice do DataFrame
    plantio_br.set_index('Data', inplace=True)
    colheita_br.set_index('Data', inplace=True)

    # Prepara os dados para o modelo
    dados_previsao = colheita_br[['Real']].copy()
    dados_previsao['Dolar'] = plantio_br['Dolar']
    dados_previsao.dropna(inplace=True)
    
    print("\n--- Dados preparados para a previsão ---")
    print(dados_previsao.head())

    # Treinamento do Modelo SARIMA
    modelo = sm.tsa.statespace.SARIMAX(dados_previsao['Real'],
                                       exog=dados_previsao['Dolar'],
                                       order=(1, 1, 1),
                                       seasonal_order=(1, 1, 1, 12),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
    
    print("\n--- Treinando o modelo SARIMA... Isso pode levar um momento. ---")
    resultado = modelo.fit(disp=False)
    print(resultado.summary())

    # Geração de Previsões Futuras (24 meses)
    # Precisamos de um DataFrame para os valores futuros da variável exógena (Dólar)
    n_previsoes = 24
    futuro_exog = pd.DataFrame({'Dolar': [dados_previsao['Dolar'].mean()] * n_previsoes}, 
                               index=pd.date_range(start=dados_previsao.index[-1] + pd.DateOffset(months=1), 
                                                   periods=n_previsoes, 
                                                   freq='MS'))
    
    previsao = resultado.get_forecast(steps=n_previsoes, exog=futuro_exog)
    
    # Intervalo de confiança da previsão
    intervalo_confianca = previsao.conf_int()

    # Visualização dos Resultados
    plt.figure(figsize=(15, 8))
    ax = dados_previsao['Real']['2020':].plot(label='Observado')
    previsao.predicted_mean.plot(ax=ax, label='Previsão')

    ax.fill_between(intervalo_confianca.index,
                    intervalo_confianca.iloc[:, 0],
                    intervalo_confianca.iloc[:, 1], color='k', alpha=.25)

    ax.set_xlabel('Data')
    ax.set_ylabel('Preço do Café (Real)')
    plt.title('Previsão do Preço do Café para os Próximos 24 Meses')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
def main():
    """
    Orquestra a execução de todas as etapas da análise.
    """
    arquivos_para_analisar = {
        "Área Plantada": "Area_Plantada.xlsx",
        "Área Colhida": "Area_Colhida.xlsx"
    }
    
    resultados_finais = {}

    for nome, caminho in arquivos_para_analisar.items():
        print(f"\nINICIANDO ANÁLISE DESCRITIVA DE: {nome}\n")
        
        dataframe_bruto = carregar_dados(caminho)
        
        if dataframe_bruto is not None:
            dataframe_limpo = limpar_dados_monetarios(dataframe_bruto)
            dados_analise, matriz_correlacao = calcular_estatisticas(dataframe_limpo, nome)
            gerar_graficos_descritivos(dados_analise, matriz_correlacao, nome)
            resultados_finais[nome] = {'df_completo': dataframe_limpo}
    
    if "Área Plantada" in resultados_finais and "Área Colhida" in resultados_finais:
        df_plantada = resultados_finais["Área Plantada"]['df_completo']
        df_colhida = resultados_finais["Área Colhida"]['df_completo']
        
        # Chama a análise comparativa e a preditiva
        analisar_producao(df_plantada, df_colhida)
        prever_precos_cafe(df_plantada, df_colhida)
    
    print("\n\nAnálise completa (Descritiva, Comparativa e Preditiva) concluída!")

# Executa a função principal quando o script é rodado
if __name__ == "__main__":
    main()