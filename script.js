const DATA_URL = "analysis_results.json";
const chartInstances = [];

const formatNumber = (value, maximumFractionDigits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits }).format(Number(value));
};

const formatDate = (value) => {
  if (!value) return "—";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("pt-BR", { dateStyle: "medium" }).format(date);
};

const setText = (id, value) => {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
};

const createElement = (tag, className, textContent) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (textContent !== undefined) element.textContent = textContent;
  return element;
};

const destroyCharts = () => {
  while (chartInstances.length) {
    const chart = chartInstances.pop();
    chart.destroy();
  }
};

const createChart = (canvasId, config) => {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  const chart = new Chart(canvas, config);
  chartInstances.push(chart);
  return chart;
};

const renderKpis = (data) => {
  const metadata = data.metadata;
  const priceModel = data.regressionModels?.priceLinear || {};
  const efficiencySeries = data.efficiencySeries || [];
  const lastEfficiency = efficiencySeries[efficiencySeries.length - 1];
  const kpis = [
    {
      label: "Registros tratados",
      value: formatNumber(metadata.totalRecords, 0),
      note: `${formatDate(metadata.startDate)} até ${formatDate(metadata.endDate)}`,
    },
    {
      label: "Localidades",
      value: formatNumber(metadata.localities, 0),
      note: `Níveis: ${(metadata.levels || []).join(", ")}`,
    },
    {
      label: "R² preço x dólar",
      value: priceModel.available ? formatNumber(priceModel.r2, 4) : "—",
      note: priceModel.available ? `${priceModel.records} observações válidas` : priceModel.message,
    },
    {
      label: "Eficiência recente",
      value: lastEfficiency ? `${formatNumber(lastEfficiency.efficiencyPct, 2)}%` : "—",
      note: lastEfficiency ? `Ano ${lastEfficiency.year}` : "Sem dados suficientes",
    },
  ];

  const kpiGrid = document.getElementById("kpiGrid");
  kpiGrid.innerHTML = "";
  kpis.forEach((item) => {
    const card = createElement("article", "kpiCard");
    card.appendChild(createElement("span", "", item.label));
    card.appendChild(createElement("strong", "", item.value));
    card.appendChild(createElement("small", "", item.note));
    kpiGrid.appendChild(card);
  });
};

const renderSummaryTable = (sourceSummaries) => {
  const body = document.getElementById("summaryTableBody");
  body.innerHTML = "";
  sourceSummaries.forEach((summary) => {
    const row = document.createElement("tr");
    [
      summary.sourceLabel,
      formatNumber(summary.records, 0),
      formatNumber(summary.localities, 0),
      formatNumber(summary.avgArabica, 2),
      formatNumber(summary.avgCanephora, 2),
      formatNumber(summary.avgReal, 2),
      `${formatNumber(summary.dataCompletenessPct, 2)}%`,
    ].forEach((value) => row.appendChild(createElement("td", "", value)));
    body.appendChild(row);
  });
};

const renderModelCards = (models) => {
  const labels = {
    priceLinear: "Regressão linear: preço x dólar",
    pricePolynomial: "Regressão polinomial: preço x dólar",
    areaLinear: "Regressão linear: plantada x colhida",
  };
  const grid = document.getElementById("modelGrid");
  grid.innerHTML = "";

  Object.entries(labels).forEach(([modelKey, label]) => {
    const model = models[modelKey] || {};
    const card = createElement("article", "modelCard");
    card.appendChild(createElement("span", "", label));
    card.appendChild(createElement("strong", "", model.available ? `R² ${formatNumber(model.r2, 4)}` : "Indisponível"));
    card.appendChild(
      createElement(
        "small",
        "",
        model.available
          ? `RMSE ${formatNumber(model.rmse, 4)} | ${formatNumber(model.records, 0)} registros`
          : model.message || "Sem dados suficientes"
      )
    );
    grid.appendChild(card);
  });
};

const renderPriceChart = (priceSeries) => {
  createChart("priceChart", {
    type: "line",
    data: {
      labels: priceSeries.map((item) => item.yearMonth),
      datasets: [
        {
          label: "Preço em R$",
          data: priceSeries.map((item) => item.real),
          borderColor: "#704214",
          backgroundColor: "rgba(112, 66, 20, 0.12)",
          tension: 0.22,
          pointRadius: 0,
        },
        {
          label: "Dólar",
          data: priceSeries.map((item) => item.dolar),
          borderColor: "#c78b3b",
          backgroundColor: "rgba(199, 139, 59, 0.12)",
          tension: 0.22,
          pointRadius: 0,
        },
      ],
    },
    options: chartOptions("R$"),
  });
};

const renderAreaChart = (annualSeries) => {
  const years = [...new Set(annualSeries.map((item) => item.year))].sort((a, b) => a - b);
  const sources = [...new Set(annualSeries.map((item) => item.sourceLabel))];
  const colors = ["#406343", "#9d6b2d"];
  const datasets = sources.map((source, index) => ({
    label: source,
    data: years.map((year) => {
      const record = annualSeries.find((item) => item.year === year && item.sourceLabel === source);
      return record?.totalCafe ?? null;
    }),
    borderColor: colors[index % colors.length],
    backgroundColor: `${colors[index % colors.length]}22`,
    tension: 0.2,
  }));

  createChart("areaChart", {
    type: "line",
    data: { labels: years, datasets },
    options: chartOptions("ha"),
  });
};

const renderEfficiencyChart = (efficiencySeries) => {
  createChart("efficiencyChart", {
    type: "line",
    data: {
      labels: efficiencySeries.map((item) => item.year),
      datasets: [
        {
          label: "Eficiência (%)",
          data: efficiencySeries.map((item) => item.efficiencyPct),
          borderColor: "#406343",
          backgroundColor: "rgba(64, 99, 67, 0.14)",
          tension: 0.22,
        },
      ],
    },
    options: chartOptions("%"),
  });
};

const renderRankingChart = (ranking) => {
  const filteredRanking = ranking.filter((item) => item.sourceLabel === "Área Colhida").slice(0, 10).reverse();
  createChart("rankingChart", {
    type: "bar",
    data: {
      labels: filteredRanking.map((item) => item.locality),
      datasets: [
        {
          label: "Área total",
          data: filteredRanking.map((item) => item.totalCafe),
          backgroundColor: "rgba(112, 66, 20, 0.78)",
          borderColor: "#704214",
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...chartOptions("ha"),
      indexAxis: "y",
    },
  });
};

const renderForecastChart = (forecast) => {
  setText("forecastMethod", `Método: ${forecast.method || forecast.message || "indisponível"}`);
  const records = forecast.records || [];
  createChart("forecastChart", {
    type: "line",
    data: {
      labels: records.map((item) => item.yearMonth),
      datasets: [
        {
          label: "Preço previsto em R$",
          data: records.map((item) => item.predictedReal),
          borderColor: "#9d3b25",
          backgroundColor: "rgba(157, 59, 37, 0.12)",
          tension: 0.22,
        },
        {
          label: "Limite inferior",
          data: records.map((item) => item.lowerReal),
          borderColor: "rgba(157, 59, 37, 0.35)",
          pointRadius: 0,
          borderDash: [6, 4],
        },
        {
          label: "Limite superior",
          data: records.map((item) => item.upperReal),
          borderColor: "rgba(157, 59, 37, 0.35)",
          pointRadius: 0,
          borderDash: [6, 4],
        },
      ],
    },
    options: chartOptions("R$"),
  });
};

const renderQuality = (quality) => {
  const container = document.getElementById("qualityList");
  container.innerHTML = "";
  (quality.columns || []).forEach((item) => {
    const wrapper = createElement("div", "qualityItem");
    const meta = createElement("div", "qualityMeta");
    meta.appendChild(createElement("span", "", item.column));
    meta.appendChild(createElement("span", "", `${formatNumber(item.validPct, 2)}%`));
    const bar = createElement("div", "progressBar");
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.validPct || 0)))}%`;
    bar.appendChild(fill);
    wrapper.appendChild(meta);
    wrapper.appendChild(bar);
    container.appendChild(wrapper);
  });
};

const renderCorrelationMatrix = (correlation) => {
  const container = document.getElementById("correlationMatrix");
  container.innerHTML = "";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(createElement("th", "", ""));
  correlation.columns.forEach((column) => headRow.appendChild(createElement("th", "", column)));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  correlation.matrix.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    tr.appendChild(createElement("th", "", correlation.columns[rowIndex]));
    row.forEach((value) => tr.appendChild(createElement("td", "", formatNumber(value, 3))));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
};

const chartOptions = (unit) => ({
  responsive: true,
  maintainAspectRatio: true,
  interaction: { intersect: false, mode: "index" },
  plugins: {
    legend: { labels: { color: "#43270d", usePointStyle: true } },
    tooltip: {
      callbacks: {
        label: (context) => {
          const parsedValue = context.chart.options.indexAxis === "y" ? context.parsed.x : context.parsed.y;
          return `${context.dataset.label}: ${formatNumber(parsedValue, 2)} ${unit}`;
        },
      },
    },
  },
  scales: {
    x: { ticks: { color: "#735f4f", maxTicksLimit: 8 }, grid: { color: "rgba(112, 66, 20, 0.08)" } },
    y: { ticks: { color: "#735f4f" }, grid: { color: "rgba(112, 66, 20, 0.08)" } },
  },
});

const renderDashboard = (data) => {
  destroyCharts();
  setText("datasetStatus", `${formatNumber(data.metadata.totalRecords, 0)} registros`);
  setText("generatedAt", `Atualizado em ${formatDate(data.metadata.generatedAt?.slice(0, 10))}`);
  renderKpis(data);
  renderSummaryTable(data.sourceSummaries || []);
  renderModelCards(data.regressionModels || {});
  renderPriceChart(data.priceSeries || []);
  renderAreaChart(data.brazilAnnualSeries || []);
  renderEfficiencyChart(data.efficiencySeries || []);
  renderRankingChart(data.latestUfRanking || []);
  renderForecastChart(data.forecast || {});
  renderQuality(data.dataQuality || {});
  renderCorrelationMatrix(data.correlationMatrix || { columns: [], matrix: [] });
};

const showLoadError = (error) => {
  setText("datasetStatus", "Erro ao carregar");
  setText("generatedAt", "Verifique analysis_results.json");
  const main = document.querySelector("main");
  const errorBox = createElement("div", "errorBox");
  errorBox.innerHTML = `<strong>Não foi possível carregar os dados.</strong><br>${error.message}<br><br>Execute <code>python Projeto_do_Cafe_no_Brasil/analise_regressao_dashboard.py --export-static</code> e publique novamente o arquivo <code>analysis_results.json</code>.`;
  main.prepend(errorBox);
};

fetch(DATA_URL, { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status} ao buscar ${DATA_URL}`);
    return response.json();
  })
  .then(renderDashboard)
  .catch(showLoadError);
