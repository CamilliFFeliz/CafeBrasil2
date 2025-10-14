import argparse
import math
import os
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ======================================================================
#                       FUNÇÕES DE UTILIDADE
# ======================================================================

def limpar_moeda_entrada(x):
    """Converte valores como 'R$ 1.234,56' em float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace('R$', '').replace('US$', '').replace('$', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(s)
    except:
        return np.nan


def carregar_dados(file_colhida='Area_Colhida.xlsx', file_plantada='Area_Plantada.xlsx'):
    """Carrega e unifica os dados."""
    dfs = []
    for f in [file_colhida, file_plantada]:
        if Path(f).exists():
            df = pd.read_excel(f)
            df.columns = [c.strip() for c in df.columns]
            if 'Data' in df.columns:
                df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            for col in ['Real', 'Dolar']:
                if col in df.columns:
                    df[col] = df[col].apply(limpar_moeda_entrada)
            for col in ['Cafe_Arabica', 'Cafe_Canephora']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("Nenhum arquivo Excel encontrado.")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['Data', 'Localidade'], keep='last')
    df = df.dropna(subset=['Data']).sort_values('Data')
    numeric_cols = [c for c in ['Cafe_Arabica', 'Cafe_Canephora', 'Real', 'Dolar'] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(lambda c: c.fillna(c.mean()))
    return df, numeric_cols


def resumo_estatistico(df, colunas):
    resultados = {}
    for col in colunas:
        s = df[col].dropna()
        resultados[col] = {
            'media': s.mean(),
            'mediana': s.median(),
            'moda': s.mode().iat[0] if not s.mode().empty else np.nan,
            'variancia': s.var(),
            'desvio_padrao': s.std(),
            'min': s.min(),
            'max': s.max(),
            'n': s.count()
        }
    return resultados

# ======================================================================
#                       FUNÇÕES DE REGRESSÃO
# ======================================================================

def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)


def ajusta_regressao_linear_simples(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    return model, y_pred, r2_score(y, y_pred), math.sqrt(mean_squared_error(y, y_pred))

def ajusta_regressao_linear_multivariada(X, y):
    """
    Ajusta uma regressão linear múltipla (multivariada).
    Retorna o modelo, previsões, R² e RMSE.
    """
    X, y = np.asarray(X), np.asarray(y)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    if len(y) < 2:
        raise ValueError("Dados insuficientes para regressão multivariada.")

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, r2_score(y, y_pred), math.sqrt(mean_squared_error(y, y_pred))



def ajusta_regressao_polinomial(x, y, grau=2):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    model = make_pipeline(PolynomialFeatures(degree=grau), LinearRegression()).fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    return model, y_pred, r2_score(y, y_pred), math.sqrt(mean_squared_error(y, y_pred))


def ajusta_exponencial(x, y):
    mask = (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    try:
        popt, _ = curve_fit(modelo_exponencial, x, y, maxfev=10000)
        y_pred = modelo_exponencial(x, *popt)
        return popt, y_pred, r2_score(y, y_pred), math.sqrt(mean_squared_error(y, y_pred))
    except:
        return None, None, None, None

# ======================================================================
#                       PREVISÃO SARIMA
# ======================================================================

def serie_mensal_padronizada(df):
    if 'Data' not in df.columns:
        raise ValueError("Coluna 'Data' não encontrada.")
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    df['Data'] = df['Data'].dt.to_period('M').dt.to_timestamp()
    df = df.groupby('Data').mean(numeric_only=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    df = df.reindex(idx)
    df = df.ffill().bfill()
    return df


def prever_sarima_plot(df, n_periods=12):
    if 'Real' not in df.columns:
        return None, "Coluna 'Real' não encontrada."
    s = df['Real'].dropna()
    if len(s) < 24:
        return None, "Dados insuficientes para SARIMA (mínimo 24 meses)."
    modelo = sm.tsa.statespace.SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    resultado = modelo.fit(disp=False)
    futuro = resultado.get_forecast(steps=n_periods)
    pred = futuro.predicted_mean
    ci = futuro.conf_int()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name='Observado'))
    fig.add_trace(go.Scatter(x=pred.index, y=pred, name='Previsão', line=dict(color='red')))
    fig.add_trace(go.Scatter(
        x=list(ci.index) + list(ci.index[::-1]),
        y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
        fill='toself', fillcolor='rgba(200,200,200,0.3)', line=dict(color='rgba(255,255,255,0)')
    ))
    fig.update_layout(title='Previsão SARIMA - Preço do Café (R$)', xaxis_title='Data', yaxis_title='Preço (R$)')
    return fig, resultado.summary().as_text()

# ======================================================================
#                       RELATÓRIO AUTOMÁTICO
# ======================================================================

def gerar_relatorio_texto(df, colunas, resultados):
    buf = StringIO()
    print("RELATÓRIO DE ANÁLISE DO CAFÉ", file=buf)
    print("="*70, file=buf)
    print(f"\nTotal de registros: {len(df)}\n", file=buf)
    print("== Estatísticas Descritivas ==\n", file=buf)
    for col in colunas:
        print(f"{col}:", file=buf)
        for k, v in resultados[col].items():
            print(f"  {k}: {v}", file=buf)
        print("", file=buf)
    print("== Correlação entre Variáveis ==\n", file=buf)
    print(df[colunas].corr(), file=buf)
    rel = buf.getvalue()
    with open("relatorio_regressao.txt", "w", encoding="utf-8") as f:
        f.write(rel)
    return rel


def gerar_graficos(df, colunas):
    os.makedirs("graficos", exist_ok=True)
    for col in colunas:
        plt.figure()
        df[col].hist(bins=20)
        plt.title(f"Distribuição - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.savefig(f"graficos/hist_{col}.png", bbox_inches='tight')
        plt.close()
    if "Real" in df.columns and "Dolar" in df.columns:
        x, y = df["Dolar"], df["Real"]
        _, y_pred, r2, _ = ajusta_regressao_linear_simples(x, y)
        plt.figure()
        plt.scatter(x, y, label="Observado")
        plt.plot(x, y_pred, color="red", label=f"Linear (R²={r2:.3f})")
        plt.title("Relação entre Dólar e Preço em Reais")
        plt.xlabel("Preço do Dólar")
        plt.ylabel("Preço do Café (R$)")
        plt.legend()
        plt.savefig("graficos/reg_preco_dolar.png", bbox_inches='tight')
        plt.close()

# ======================================================================
#                       DASHBOARD STREAMLIT
# ======================================================================

def streamlit_app(df, numeric_cols):
    st.set_page_config(page_title="Análise e Regressão - Café", layout="wide")
    st.title("📈 Dashboard: Análises e Modelos de Regressão do Café")

    st.sidebar.header("Configurações")
    target = st.sidebar.selectbox("Variável dependente (y)", numeric_cols, index=len(numeric_cols) - 1)
    features = st.sidebar.multiselect("Variáveis independentes (X)", [c for c in numeric_cols if c != target],
                                      default=[c for c in numeric_cols if c != target])
    grau_poly = st.sidebar.slider("Grau do polinômio", 2, 5, 2)

    stats = resumo_estatistico(df, numeric_cols)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Estatísticas",
    "📈 Distribuições",
    "🔗 Modelos",
    "⏱ Série Temporal",
    "🖼️ Infografico"
])

    # ===== TAB 1 - Estatísticas =====
    with tab1:
        st.write(pd.DataFrame.from_dict(stats, orient='index'))
        st.subheader("📈 Gráficos Estatísticos")

        # Converter para DataFrame para facilitar plotagem
        df_stats = pd.DataFrame(stats).T

        # Gráfico de barras das medidas centrais
        st.plotly_chart(
            px.bar(df_stats[['media', 'mediana', 'moda']],
                   barmode='group',
                   title="Média, Mediana e Moda por Variável"),
            use_container_width=True
        )

        # Gráfico de variância
        st.plotly_chart(
            px.bar(df_stats[['variancia']],
                   title="Variância por Variável"),
            use_container_width=True
        )

        # Gráfico de desvio padrão
        st.plotly_chart(
            px.bar(df_stats[['desvio_padrao']],
                   title="Desvio Padrão por Variável"),
            use_container_width=True
        )

        # Gráfico de mínimos e máximos
        st.plotly_chart(
            px.bar(df_stats[['min', 'max']],
                   barmode='group',
                   title="Valores Mínimos e Máximos por Variável"),
            use_container_width=True
        )

        # Gráfico de contagem (N)
        st.plotly_chart(
            px.bar(df_stats[['n']],
                   title="Número de Registros (N) por Variável"),
            use_container_width=True
        )

        # Gráfico de correlação entre variáveis numéricas
        st.subheader("🔗 Matriz de Correlação")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             title="Correlação entre Variáveis")
        st.plotly_chart(fig_corr, use_container_width=True)


    # ===== TAB 2 - Distribuições =====
    with tab2:
        for col in numeric_cols:
            st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Distribuição - {col}"), use_container_width=True)

    # ===== TAB 3 - Modelos =====
    with tab3:
        if len(features) >= 1:
            xcol, ycol = features[0], target
            x, y = df[xcol].values, df[ycol].values

            # Regressão Linear Simples
            model_lin, y_pred_lin, r2_lin, rmse_lin = ajusta_regressao_linear_simples(x, y)
            # Regressão Polinomial
            model_poly, y_pred_poly, r2_poly, rmse_poly = ajusta_regressao_polinomial(x, y, grau_poly)
            # Exponencial
            popt, y_pred_exp, r2_exp, rmse_exp = ajusta_exponencial(x, y)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Observado'))
            fig.add_trace(go.Scatter(x=x, y=y_pred_lin, mode='lines', name=f'Linear (R²={r2_lin:.3f})'))
            fig.add_trace(go.Scatter(x=x, y=y_pred_poly, mode='lines', name=f'Polinomial (R²={r2_poly:.3f})'))
            if y_pred_exp is not None:
                fig.add_trace(go.Scatter(x=x, y=y_pred_exp, mode='lines', name=f'Exponencial (R²={r2_exp:.3f})'))
            fig.update_layout(title=f"{ycol} vs {xcol}", xaxis_title=xcol, yaxis_title=ycol)
            st.plotly_chart(fig, use_container_width=True)

            st.write(pd.DataFrame({
                "Modelo": ["Linear", f"Polinomial (grau {grau_poly})", "Exponencial"],
                "R²": [r2_lin, r2_poly, r2_exp],
                "RMSE": [rmse_lin, rmse_poly, rmse_exp]
            }))

        # Regressão Multivariada
        if len(features) >= 2:
            X, y = df[features].values, df[target].values
            model_mv, y_pred_mv, r2_mv, rmse_mv = ajusta_regressao_linear_multivariada(X, y)
            st.subheader("Regressão Linear Multivariada")
            st.write(pd.DataFrame({"Variável": features, "Coeficiente": model_mv.coef_}))
            st.write(f"Intercepto: {model_mv.intercept_:.3f}")
            st.write(f"R²: {r2_mv:.3f} | RMSE: {rmse_mv:.3f}")

    # ===== TAB 4 - Série Temporal =====
    with tab4:
        df_ts = serie_mensal_padronizada(df)
        fig_sarima, resumo_sarima = prever_sarima_plot(df_ts)
        if fig_sarima:
            st.plotly_chart(fig_sarima, use_container_width=True)
            st.text(resumo_sarima)
        else:
            st.warning(resumo_sarima)

        # ===== TAB 5 - Imagem =====
    with tab5:
        st.header("🖼️ Exibição do Infografico")
        imagem_path = "Info.jpeg"  # caminho da imagem a ser exibida
        if os.path.exists(imagem_path):
            st.image(imagem_path, use_container_width=True)
        else:
            st.warning(f"Imagem '{imagem_path}' não encontrada. Coloque o arquivo na mesma pasta do script.")



# ======================================================================
#                       MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help="Gera relatório e gráficos automaticamente")
    parser.add_argument('--file_colhida', default='Area_Colhida.xlsx')
    parser.add_argument('--file_plantada', default='Area_Plantada.xlsx')
    args = parser.parse_args()

    df, numeric_cols = carregar_dados(args.file_colhida, args.file_plantada)
    resultados = resumo_estatistico(df, numeric_cols)

    if args.report:
        print("🧾 Gerando relatório e gráficos...")
        gerar_graficos(df, numeric_cols)
        gerar_relatorio_texto(df, numeric_cols, resultados)
        print("✅ Relatório salvo como relatorio_regressao.txt")
        print("✅ Gráficos salvos na pasta 'graficos/'")
    else:
        streamlit_app(df, numeric_cols)


if __name__ == "__main__":
    main()
