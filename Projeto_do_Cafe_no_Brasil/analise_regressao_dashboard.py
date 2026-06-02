from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda value: f"{value:.2f}")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_HARVEST_FILE = SCRIPT_DIR / "Area_Colhida.xlsx"
DEFAULT_PLANTED_FILE = SCRIPT_DIR / "Area_Plantada.xlsx"
DEFAULT_STATIC_JSON_PATH = PROJECT_ROOT / "analysis_results.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "relatorio_regressao.txt"
DEFAULT_CHARTS_DIR = PROJECT_ROOT / "graficos"
INFOGRAPHIC_PATH = PROJECT_ROOT / "assets" / "info.jpeg"

NUMERIC_COLUMNS = ("Cafe_Arabica", "Cafe_Canephora", "Real", "Dolar")
AREA_COLUMNS = ("Cafe_Arabica", "Cafe_Canephora", "TotalCafe")
SOURCE_CONFIGS = (
    {
        "sourceKey": "areaColhida",
        "sourceLabel": "Área Colhida",
        "defaultPath": DEFAULT_HARVEST_FILE,
    },
    {
        "sourceKey": "areaPlantada",
        "sourceLabel": "Área Plantada",
        "defaultPath": DEFAULT_PLANTED_FILE,
    },
)
MIN_REGRESSION_ROWS = 3
FORECAST_PERIODS = 12
REQUIRED_COLUMNS = ("Nivel", "Cod", "Localidade", "Data")


def normalizePath(filePath: str | Path | None, defaultPath: Path) -> Path:
    if filePath is None:
        return defaultPath
    candidatePath = Path(filePath)
    if candidatePath.is_absolute():
        return candidatePath
    cwdCandidatePath = Path.cwd() / candidatePath
    if cwdCandidatePath.exists():
        return cwdCandidatePath
    return SCRIPT_DIR / candidatePath


def cleanCurrencyValue(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    valueText = str(value).strip()
    if not valueText:
        return np.nan

    normalizedText = re.sub(r"[^0-9,.-]", "", valueText)
    if not normalizedText:
        return np.nan

    if "," in normalizedText and "." in normalizedText:
        normalizedText = normalizedText.replace(".", "").replace(",", ".")
    elif "," in normalizedText:
        normalizedText = normalizedText.replace(",", ".")

    try:
        return float(normalizedText)
    except ValueError:
        return np.nan


def safeFloat(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numericValue = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numericValue):
        return None
    return round(numericValue, digits)


def safeInt(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def formatDate(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def validateColumns(dataFrame: pd.DataFrame, filePath: Path) -> None:
    missingColumns = [column for column in REQUIRED_COLUMNS if column not in dataFrame.columns]
    if missingColumns:
        joinedColumns = ", ".join(missingColumns)
        raise ValueError(f"Arquivo {filePath.name} sem colunas obrigatórias: {joinedColumns}")


def loadWorkbook(filePath: Path, sourceKey: str, sourceLabel: str) -> pd.DataFrame:
    if not filePath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filePath}")

    dataFrame = pd.read_excel(filePath, sheet_name=0)
    dataFrame.columns = [str(column).strip() for column in dataFrame.columns]
    validateColumns(dataFrame, filePath)

    dataFrame["Data"] = pd.to_datetime(dataFrame["Data"], errors="coerce")
    dataFrame["Localidade"] = dataFrame["Localidade"].astype(str).str.strip()
    dataFrame["Nivel"] = dataFrame["Nivel"].astype(str).str.strip().str.upper()

    for column in ("Real", "Dolar"):
        if column in dataFrame.columns:
            dataFrame[column] = dataFrame[column].apply(cleanCurrencyValue)

    for column in ("Cafe_Arabica", "Cafe_Canephora"):
        if column in dataFrame.columns:
            dataFrame[column] = pd.to_numeric(dataFrame[column], errors="coerce")

    dataFrame = dataFrame.dropna(subset=["Data"])
    dataFrame = dataFrame[dataFrame["Localidade"].ne("")]
    dataFrame["SourceKey"] = sourceKey
    dataFrame["SourceLabel"] = sourceLabel
    dataFrame["Ano"] = dataFrame["Data"].dt.year.astype(int)
    dataFrame["Mes"] = dataFrame["Data"].dt.month.astype(int)
    dataFrame["AnoMes"] = dataFrame["Data"].dt.strftime("%Y-%m")
    dataFrame["TotalCafe"] = dataFrame[["Cafe_Arabica", "Cafe_Canephora"]].sum(axis=1, min_count=1)

    return dataFrame.sort_values(["Data", "SourceKey", "Nivel", "Localidade"]).reset_index(drop=True)


def loadData(fileColhida: str | Path | None = None, filePlantada: str | Path | None = None) -> pd.DataFrame:
    filePathBySource = {
        "areaColhida": normalizePath(fileColhida, DEFAULT_HARVEST_FILE),
        "areaPlantada": normalizePath(filePlantada, DEFAULT_PLANTED_FILE),
    }

    dataFrames: list[pd.DataFrame] = []
    for sourceConfig in SOURCE_CONFIGS:
        dataFrames.append(
            loadWorkbook(
                filePathBySource[sourceConfig["sourceKey"]],
                sourceConfig["sourceKey"],
                sourceConfig["sourceLabel"],
            )
        )

    combinedData = pd.concat(dataFrames, ignore_index=True)
    combinedData = combinedData.drop_duplicates(
        subset=["SourceKey", "Data", "Nivel", "Cod", "Localidade"],
        keep="last",
    )
    return combinedData.sort_values(["Data", "SourceKey", "Nivel", "Cod", "Localidade"]).reset_index(drop=True)


def getBrazilData(dataFrame: pd.DataFrame) -> pd.DataFrame:
    brazilData = dataFrame[dataFrame["Nivel"].eq("BR")].copy()
    if brazilData.empty:
        return dataFrame[dataFrame["Localidade"].str.upper().eq("BRASIL")].copy()
    return brazilData


def getValidNumericColumns(dataFrame: pd.DataFrame) -> list[str]:
    return [column for column in (*NUMERIC_COLUMNS, "TotalCafe") if column in dataFrame.columns and dataFrame[column].notna().any()]


def calculateDescriptiveStats(dataFrame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    stats: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        series = pd.to_numeric(dataFrame[column], errors="coerce").dropna()
        modeSeries = series.mode()
        stats[column] = {
            "media": safeFloat(series.mean()),
            "mediana": safeFloat(series.median()),
            "moda": safeFloat(modeSeries.iloc[0]) if not modeSeries.empty else None,
            "variancia": safeFloat(series.var()),
            "desvioPadrao": safeFloat(series.std()),
            "minimo": safeFloat(series.min()),
            "maximo": safeFloat(series.max()),
            "registrosValidos": safeInt(series.count()),
        }
    return stats


def buildSourceSummaries(dataFrame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for sourceLabel, sourceData in dataFrame.groupby("SourceLabel", sort=True):
        brazilData = getBrazilData(sourceData)
        areaBaseData = brazilData if not brazilData.empty else sourceData
        summaries.append(
            {
                "sourceLabel": sourceLabel,
                "records": safeInt(len(sourceData)),
                "validBrazilRecords": safeInt(len(brazilData)),
                "localities": safeInt(sourceData["Localidade"].nunique()),
                "startDate": formatDate(sourceData["Data"].min()),
                "endDate": formatDate(sourceData["Data"].max()),
                "avgArabica": safeFloat(areaBaseData["Cafe_Arabica"].mean()),
                "avgCanephora": safeFloat(areaBaseData["Cafe_Canephora"].mean()),
                "avgTotalCafe": safeFloat(areaBaseData["TotalCafe"].mean()),
                "avgReal": safeFloat(sourceData["Real"].mean()),
                "avgDolar": safeFloat(sourceData["Dolar"].mean()),
                "dataCompletenessPct": safeFloat(100 * sourceData[list(NUMERIC_COLUMNS)].notna().mean().mean(), 2),
            }
        )
    return summaries


def buildBrazilAnnualSeries(dataFrame: pd.DataFrame) -> list[dict[str, Any]]:
    brazilData = getBrazilData(dataFrame)
    if brazilData.empty:
        return []

    groupedData = (
        brazilData.groupby(["Ano", "SourceLabel"], as_index=False)
        .agg(
            Cafe_Arabica=("Cafe_Arabica", "mean"),
            Cafe_Canephora=("Cafe_Canephora", "mean"),
            TotalCafe=("TotalCafe", "mean"),
            Real=("Real", "mean"),
            Dolar=("Dolar", "mean"),
            registros=("Data", "count"),
        )
        .sort_values(["Ano", "SourceLabel"])
    )

    records: list[dict[str, Any]] = []
    for row in groupedData.to_dict("records"):
        records.append(
            {
                "year": safeInt(row["Ano"]),
                "sourceLabel": row["SourceLabel"],
                "arabica": safeFloat(row["Cafe_Arabica"]),
                "canephora": safeFloat(row["Cafe_Canephora"]),
                "totalCafe": safeFloat(row["TotalCafe"]),
                "real": safeFloat(row["Real"]),
                "dolar": safeFloat(row["Dolar"]),
                "records": safeInt(row["registros"]),
            }
        )
    return records


def buildPriceSeries(dataFrame: pd.DataFrame) -> list[dict[str, Any]]:
    monthlyData = dataFrame.copy()
    monthlyData["Data"] = monthlyData["Data"].dt.to_period("M").dt.to_timestamp()
    priceData = (
        monthlyData.groupby("Data", as_index=False)
        .agg(Real=("Real", "mean"), Dolar=("Dolar", "mean"))
        .dropna(subset=["Real"])
        .sort_values("Data")
    )

    return [
        {
            "date": formatDate(row["Data"]),
            "yearMonth": pd.Timestamp(row["Data"]).strftime("%Y-%m"),
            "real": safeFloat(row["Real"]),
            "dolar": safeFloat(row["Dolar"]),
        }
        for row in priceData.to_dict("records")
    ]


def buildLatestUfRanking(dataFrame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    ufData = dataFrame[dataFrame["Nivel"].eq("UF")].copy()
    if ufData.empty:
        return []

    rankingRecords: list[dict[str, Any]] = []
    for sourceLabel, sourceData in ufData.groupby("SourceLabel"):
        latestDate = sourceData["Data"].max()
        latestData = sourceData[sourceData["Data"].eq(latestDate)].copy()
        latestData = latestData.sort_values("TotalCafe", ascending=False).head(limit)
        for row in latestData.to_dict("records"):
            rankingRecords.append(
                {
                    "sourceLabel": sourceLabel,
                    "date": formatDate(row["Data"]),
                    "locality": row["Localidade"],
                    "arabica": safeFloat(row["Cafe_Arabica"]),
                    "canephora": safeFloat(row["Cafe_Canephora"]),
                    "totalCafe": safeFloat(row["TotalCafe"]),
                }
            )
    return rankingRecords


def buildEfficiencySeries(dataFrame: pd.DataFrame) -> list[dict[str, Any]]:
    brazilData = getBrazilData(dataFrame)
    if brazilData.empty:
        return []

    pivotData = brazilData.pivot_table(
        index="Data",
        columns="SourceLabel",
        values="TotalCafe",
        aggfunc="mean",
    ).reset_index()

    if "Área Colhida" not in pivotData.columns or "Área Plantada" not in pivotData.columns:
        return []

    pivotData["efficiencyPct"] = np.where(
        pivotData["Área Plantada"].gt(0),
        100 * pivotData["Área Colhida"] / pivotData["Área Plantada"],
        np.nan,
    )
    pivotData["Ano"] = pivotData["Data"].dt.year
    annualData = (
        pivotData.groupby("Ano", as_index=False)
        .agg(
            areaColhida=("Área Colhida", "mean"),
            areaPlantada=("Área Plantada", "mean"),
            efficiencyPct=("efficiencyPct", "mean"),
        )
        .sort_values("Ano")
    )

    return [
        {
            "year": safeInt(row["Ano"]),
            "areaColhida": safeFloat(row["areaColhida"]),
            "areaPlantada": safeFloat(row["areaPlantada"]),
            "efficiencyPct": safeFloat(row["efficiencyPct"], 2),
        }
        for row in annualData.to_dict("records")
    ]


def buildCorrelationMatrix(dataFrame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if len(columns) < 2:
        return {"columns": columns, "matrix": []}

    correlationData = dataFrame[columns].corr(numeric_only=True)
    matrix = []
    for rowColumn in correlationData.index:
        matrixRow = []
        for column in correlationData.columns:
            matrixRow.append(safeFloat(correlationData.loc[rowColumn, column], 4))
        matrix.append(matrixRow)

    return {"columns": list(correlationData.columns), "matrix": matrix}


def fitLinearModel(xValues: pd.Series | np.ndarray, yValues: pd.Series | np.ndarray) -> dict[str, Any]:
    xArray = np.asarray(xValues, dtype=float).reshape(-1, 1)
    yArray = np.asarray(yValues, dtype=float)
    validMask = np.isfinite(xArray).ravel() & np.isfinite(yArray)
    xClean = xArray[validMask]
    yClean = yArray[validMask]

    if len(yClean) < MIN_REGRESSION_ROWS:
        return {"available": False, "message": "Dados insuficientes para regressão linear."}

    model = LinearRegression().fit(xClean, yClean)
    predictions = model.predict(xClean)
    return {
        "available": True,
        "records": safeInt(len(yClean)),
        "r2": safeFloat(r2_score(yClean, predictions), 4),
        "rmse": safeFloat(math.sqrt(mean_squared_error(yClean, predictions)), 4),
        "intercept": safeFloat(model.intercept_, 4),
        "coefficient": safeFloat(model.coef_[0], 4),
    }


def fitPolynomialModel(xValues: pd.Series | np.ndarray, yValues: pd.Series | np.ndarray, degree: int = 2) -> dict[str, Any]:
    xArray = np.asarray(xValues, dtype=float).reshape(-1, 1)
    yArray = np.asarray(yValues, dtype=float)
    validMask = np.isfinite(xArray).ravel() & np.isfinite(yArray)
    xClean = xArray[validMask]
    yClean = yArray[validMask]

    if len(yClean) < MIN_REGRESSION_ROWS:
        return {"available": False, "message": "Dados insuficientes para regressão polinomial."}

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(xClean, yClean)
    predictions = model.predict(xClean)
    return {
        "available": True,
        "records": safeInt(len(yClean)),
        "degree": degree,
        "r2": safeFloat(r2_score(yClean, predictions), 4),
        "rmse": safeFloat(math.sqrt(mean_squared_error(yClean, predictions)), 4),
    }


def buildRegressionModels(dataFrame: pd.DataFrame) -> dict[str, Any]:
    priceSeries = pd.DataFrame(buildPriceSeries(dataFrame))
    priceModel = {"available": False, "message": "Série de preço indisponível."}
    pricePolynomialModel = {"available": False, "message": "Série de preço indisponível."}
    if not priceSeries.empty and {"real", "dolar"}.issubset(priceSeries.columns):
        priceModel = fitLinearModel(priceSeries["dolar"], priceSeries["real"])
        pricePolynomialModel = fitPolynomialModel(priceSeries["dolar"], priceSeries["real"], degree=2)

    brazilData = getBrazilData(dataFrame)
    pivotData = brazilData.pivot_table(
        index="Data",
        columns="SourceLabel",
        values="TotalCafe",
        aggfunc="mean",
    ).reset_index()
    areaModel = {"available": False, "message": "Comparação entre área plantada e colhida indisponível."}
    if {"Área Plantada", "Área Colhida"}.issubset(pivotData.columns):
        areaModel = fitLinearModel(pivotData["Área Plantada"], pivotData["Área Colhida"])

    return {
        "priceLinear": priceModel,
        "pricePolynomial": pricePolynomialModel,
        "areaLinear": areaModel,
    }


def buildForecast(priceSeries: list[dict[str, Any]], periods: int = FORECAST_PERIODS) -> dict[str, Any]:
    if not priceSeries:
        return {"available": False, "message": "Série de preço indisponível.", "records": []}

    priceData = pd.DataFrame(priceSeries)
    priceData["date"] = pd.to_datetime(priceData["date"], errors="coerce")
    priceData = priceData.dropna(subset=["date", "real"]).sort_values("date")
    if len(priceData) < 24:
        return {"available": False, "message": "Dados insuficientes para previsão mensal.", "records": []}

    try:
        import statsmodels.api as sm

        timeSeries = priceData.set_index("date")["real"].asfreq("MS")
        timeSeries = timeSeries.ffill().bfill()
        model = sm.tsa.statespace.SARIMAX(
            timeSeries,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=periods)
        confidenceInterval = forecast.conf_int()
        records = []
        for dateValue, predictedValue in forecast.predicted_mean.items():
            records.append(
                {
                    "date": formatDate(dateValue),
                    "yearMonth": pd.Timestamp(dateValue).strftime("%Y-%m"),
                    "predictedReal": safeFloat(predictedValue),
                    "lowerReal": safeFloat(confidenceInterval.loc[dateValue].iloc[0]),
                    "upperReal": safeFloat(confidenceInterval.loc[dateValue].iloc[1]),
                }
            )
        return {"available": True, "method": "SARIMA(1,1,1)(1,1,1,12)", "records": records}
    except Exception as error:
        priceData["periodIndex"] = np.arange(len(priceData))
        model = LinearRegression().fit(priceData[["periodIndex"]], priceData["real"])
        futureIndex = np.arange(len(priceData), len(priceData) + periods)
        futureDates = pd.date_range(priceData["date"].max() + pd.DateOffset(months=1), periods=periods, freq="MS")
        futurePredictions = model.predict(futureIndex.reshape(-1, 1))
        records = [
            {
                "date": formatDate(dateValue),
                "yearMonth": pd.Timestamp(dateValue).strftime("%Y-%m"),
                "predictedReal": safeFloat(predictedValue),
                "lowerReal": None,
                "upperReal": None,
            }
            for dateValue, predictedValue in zip(futureDates, futurePredictions)
        ]
        return {
            "available": True,
            "method": "Tendência linear mensal; fallback por falha no SARIMA",
            "warning": str(error),
            "records": records,
        }


def buildDataQuality(dataFrame: pd.DataFrame) -> dict[str, Any]:
    qualityByColumn = []
    for column in [*REQUIRED_COLUMNS, *NUMERIC_COLUMNS, "TotalCafe"]:
        if column not in dataFrame.columns:
            continue
        validCount = safeInt(dataFrame[column].notna().sum())
        qualityByColumn.append(
            {
                "column": column,
                "validRecords": validCount,
                "missingRecords": safeInt(dataFrame[column].isna().sum()),
                "validPct": safeFloat(100 * validCount / max(len(dataFrame), 1), 2),
            }
        )

    duplicateCount = safeInt(
        dataFrame.duplicated(subset=["SourceKey", "Data", "Nivel", "Cod", "Localidade"]).sum()
    )
    return {"duplicateRecords": duplicateCount, "columns": qualityByColumn}


def buildAnalysisPayload(dataFrame: pd.DataFrame) -> dict[str, Any]:
    numericColumns = getValidNumericColumns(dataFrame)
    priceSeries = buildPriceSeries(dataFrame)
    payload = {
        "metadata": {
            "generatedAt": datetime.now().isoformat(timespec="seconds"),
            "projectName": "Café Brasil - Análise de Dados",
            "totalRecords": safeInt(len(dataFrame)),
            "startDate": formatDate(dataFrame["Data"].min()),
            "endDate": formatDate(dataFrame["Data"].max()),
            "localities": safeInt(dataFrame["Localidade"].nunique()),
            "levels": sorted(dataFrame["Nivel"].dropna().unique().tolist()),
            "sources": [sourceConfig["sourceLabel"] for sourceConfig in SOURCE_CONFIGS],
        },
        "sourceSummaries": buildSourceSummaries(dataFrame),
        "descriptiveStats": calculateDescriptiveStats(dataFrame, numericColumns),
        "brazilAnnualSeries": buildBrazilAnnualSeries(dataFrame),
        "priceSeries": priceSeries,
        "latestUfRanking": buildLatestUfRanking(dataFrame),
        "efficiencySeries": buildEfficiencySeries(dataFrame),
        "correlationMatrix": buildCorrelationMatrix(dataFrame, numericColumns),
        "regressionModels": buildRegressionModels(dataFrame),
        "forecast": buildForecast(priceSeries),
        "dataQuality": buildDataQuality(dataFrame),
    }
    return payload


def exportStaticJson(payload: dict[str, Any], outputPath: Path = DEFAULT_STATIC_JSON_PATH) -> Path:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputPath


def generateReportText(payload: dict[str, Any]) -> str:
    buffer = StringIO()
    metadata = payload["metadata"]
    print("RELATÓRIO DE ANÁLISE DO CAFÉ BRASIL", file=buffer)
    print("=" * 72, file=buffer)
    print(f"Gerado em: {metadata['generatedAt']}", file=buffer)
    print(f"Registros analisados: {metadata['totalRecords']}", file=buffer)
    print(f"Período: {metadata['startDate']} até {metadata['endDate']}", file=buffer)
    print(f"Localidades: {metadata['localities']}", file=buffer)
    print("\nRESUMO POR BASE", file=buffer)
    print("-" * 72, file=buffer)
    for summary in payload["sourceSummaries"]:
        print(
            f"{summary['sourceLabel']}: {summary['records']} registros; "
            f"média Brasil TotalCafe={summary['avgTotalCafe']}; "
            f"preço médio Real={summary['avgReal']}; dólar médio={summary['avgDolar']}.",
            file=buffer,
        )

    print("\nMODELOS", file=buffer)
    print("-" * 72, file=buffer)
    for modelName, modelInfo in payload["regressionModels"].items():
        if modelInfo.get("available"):
            print(
                f"{modelName}: R²={modelInfo.get('r2')}; RMSE={modelInfo.get('rmse')}; "
                f"coeficiente={modelInfo.get('coefficient')}; registros={modelInfo.get('records')}.",
                file=buffer,
            )
        else:
            print(f"{modelName}: {modelInfo.get('message')}", file=buffer)

    print("\nQUALIDADE DOS DADOS", file=buffer)
    print("-" * 72, file=buffer)
    print(f"Duplicatas remanescentes: {payload['dataQuality']['duplicateRecords']}", file=buffer)
    for columnInfo in payload["dataQuality"]["columns"]:
        print(
            f"{columnInfo['column']}: {columnInfo['validPct']}% válidos "
            f"({columnInfo['validRecords']} válidos / {columnInfo['missingRecords']} ausentes)",
            file=buffer,
        )

    return buffer.getvalue()


def exportReport(payload: dict[str, Any], reportPath: Path = DEFAULT_REPORT_PATH) -> Path:
    reportPath.write_text(generateReportText(payload), encoding="utf-8")
    return reportPath


def generateStaticCharts(payload: dict[str, Any], chartsDir: Path = DEFAULT_CHARTS_DIR) -> Path:
    chartsDir.mkdir(parents=True, exist_ok=True)

    priceData = pd.DataFrame(payload["priceSeries"])
    if not priceData.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(priceData["date"]), priceData["real"], label="Preço em R$ observado")
        plt.title("Série histórica do preço do café em reais")
        plt.xlabel("Data")
        plt.ylabel("Preço em R$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(chartsDir / "serie_preco_real.png", dpi=150)
        plt.close()

    efficiencyData = pd.DataFrame(payload["efficiencySeries"])
    if not efficiencyData.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(efficiencyData["year"], efficiencyData["efficiencyPct"], marker="o")
        plt.title("Eficiência nacional: área colhida / área plantada")
        plt.xlabel("Ano")
        plt.ylabel("Eficiência (%)")
        plt.tight_layout()
        plt.savefig(chartsDir / "eficiencia_area.png", dpi=150)
        plt.close()

    return chartsDir


def runStreamlitApp(payload: dict[str, Any]) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as error:
        raise RuntimeError(
            "Streamlit/Plotly não estão instalados. Execute: pip install -r requirements.txt"
        ) from error

    st.set_page_config(page_title="Café Brasil - Análise de Dados", layout="wide")
    st.title("Café Brasil - Dashboard Analítico")
    st.caption("Análise consolidada de área plantada, área colhida, preços em reais e dólar.")

    metadata = payload["metadata"]
    kpiColumns = st.columns(4)
    kpiColumns[0].metric("Registros", f"{metadata['totalRecords']:,}".replace(",", "."))
    kpiColumns[1].metric("Localidades", metadata["localities"])
    kpiColumns[2].metric("Início", metadata["startDate"])
    kpiColumns[3].metric("Fim", metadata["endDate"])

    tabOverview, tabSeries, tabModels, tabQuality, tabInfographic = st.tabs(
        ["Visão geral", "Séries", "Modelos", "Qualidade", "Infográfico"]
    )

    with tabOverview:
        st.subheader("Resumo por fonte")
        st.dataframe(pd.DataFrame(payload["sourceSummaries"]), use_container_width=True)
        st.subheader("Estatísticas descritivas")
        st.dataframe(pd.DataFrame(payload["descriptiveStats"]).T, use_container_width=True)

    with tabSeries:
        priceData = pd.DataFrame(payload["priceSeries"])
        if not priceData.empty:
            priceData["date"] = pd.to_datetime(priceData["date"])
            st.plotly_chart(
                px.line(priceData, x="date", y=["real", "dolar"], title="Preço do café e dólar"),
                use_container_width=True,
            )

        annualData = pd.DataFrame(payload["brazilAnnualSeries"])
        if not annualData.empty:
            st.plotly_chart(
                px.line(
                    annualData,
                    x="year",
                    y="totalCafe",
                    color="sourceLabel",
                    markers=True,
                    title="Área média nacional por ano",
                ),
                use_container_width=True,
            )

    with tabModels:
        modelRows = []
        for modelName, modelInfo in payload["regressionModels"].items():
            row = {"modelo": modelName, **modelInfo}
            modelRows.append(row)
        st.dataframe(pd.DataFrame(modelRows), use_container_width=True)

        forecastData = pd.DataFrame(payload["forecast"].get("records", []))
        if not forecastData.empty:
            forecastData["date"] = pd.to_datetime(forecastData["date"])
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=forecastData["date"], y=forecastData["predictedReal"], name="Previsão"))
            figure.update_layout(title=f"Previsão de preço - {payload['forecast'].get('method')}")
            st.plotly_chart(figure, use_container_width=True)

    with tabQuality:
        st.metric("Duplicatas remanescentes", payload["dataQuality"]["duplicateRecords"])
        st.dataframe(pd.DataFrame(payload["dataQuality"]["columns"]), use_container_width=True)

    with tabInfographic:
        if INFOGRAPHIC_PATH.exists():
            st.image(str(INFOGRAPHIC_PATH), use_container_width=True)
        else:
            st.warning("Infográfico não encontrado em assets/info.jpeg.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline analítico do projeto Café Brasil")
    parser.add_argument("--report", action="store_true", help="Gera relatório TXT, JSON estático e gráficos PNG.")
    parser.add_argument("--export-static", action="store_true", help="Gera apenas o arquivo analysis_results.json.")
    parser.add_argument("--file-colhida", default=None, help="Caminho alternativo para Area_Colhida.xlsx.")
    parser.add_argument("--file-plantada", default=None, help="Caminho alternativo para Area_Plantada.xlsx.")
    args = parser.parse_args()

    dataFrame = loadData(args.file_colhida, args.file_plantada)
    payload = buildAnalysisPayload(dataFrame)

    if args.report:
        jsonPath = exportStaticJson(payload)
        reportPath = exportReport(payload)
        chartsDir = generateStaticCharts(payload)
        print(f"JSON estático gerado: {jsonPath}")
        print(f"Relatório gerado: {reportPath}")
        print(f"Gráficos gerados em: {chartsDir}")
        return

    if args.export_static:
        jsonPath = exportStaticJson(payload)
        print(f"JSON estático gerado: {jsonPath}")
        return

    runStreamlitApp(payload)


if __name__ == "__main__":
    main()
