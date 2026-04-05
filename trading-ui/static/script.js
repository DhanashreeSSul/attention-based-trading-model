/* AlphaSignal — script.js */

let priceChart = null;

// ── Default dates (last 90 days) ──────────
(function initDates() {
  const today = new Date();
  const prior = new Date();
  prior.setDate(today.getDate() - 90);
  document.getElementById("endDate").value   = toISO(today);
  document.getElementById("startDate").value = toISO(prior);
})();

function toISO(d) {
  return d.toISOString().split("T")[0];
}

// ── Quick-ticker shortcut ─────────────────
function setTicker(sym) {
  document.getElementById("ticker").value = sym;
}

// ── UI helpers ────────────────────────────
function show(id)       { document.getElementById(id).style.display = "block"; }
function hide(id)       { document.getElementById(id).style.display = "none";  }
function showFlex(id)   { document.getElementById(id).style.display = "flex";  }
function setText(id, v) { document.getElementById(id).textContent = v; }
function setClass(id, cls) {
  const el = document.getElementById(id);
  el.className = el.className.replace(/\bup\b|\bdown\b|\bamber\b/g, "").trim();
  if (cls) el.classList.add(cls);
}

// ── Main prediction ────────────────────────
async function runPrediction() {
  const ticker    = document.getElementById("ticker").value.trim().toUpperCase();
  const startDate = document.getElementById("startDate").value;
  const endDate   = document.getElementById("endDate").value;

  if (!ticker) { showError("Please enter a stock ticker symbol."); return; }
  if (startDate && endDate && startDate >= endDate) {
    showError("Start date must be before end date."); return;
  }

  // Reset UI
  hide("errorBox");
  hide("demoBox");
  hide("results");
  show("spinnerWrap");
  document.getElementById("predictBtn").disabled = true;

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate }),
    });
    const data = await res.json();

    hide("spinnerWrap");

    if (!res.ok) {
      showError(data.error || "Prediction failed. Try another ticker.");
      return;
    }

    renderResults(data);

  } catch (err) {
    hide("spinnerWrap");
    showError("Could not connect to the server. Make sure Flask is running on port 5000.");
  } finally {
    document.getElementById("predictBtn").disabled = false;
  }
}

// ── Render results ─────────────────────────
function renderResults(d) {
  const isUp = d.direction === "UP";
  const cls  = isUp ? "up" : "down";

  // Demo notice
  if (d.demo_mode) show("demoBox");

  // Signal direction
  const sigEl = document.getElementById("signalDirection");
  sigEl.textContent = d.direction;
  setClass("signalDirection", cls);

  // Confidence bar
  const bar = document.getElementById("confBar");
  bar.className = "conf-bar";
  bar.classList.add(cls);
  setTimeout(() => { bar.style.width = d.confidence + "%"; }, 50);

  setText("confValue", d.confidence + "%");

  // Suggestion
  const sugEl = document.getElementById("suggestionValue");
  sugEl.textContent = d.suggestion;
  setClass("suggestionValue", cls);

  // Stats
  setText("statTicker", d.ticker);
  document.getElementById("statConf").textContent = d.confidence + "%";
  setClass("statConf", cls);
  setText("statAcc",   d.accuracy + "%");
  setText("statRange", d.start_date + " → " + d.end_date);

  // Chart
  renderChart(d.chart, d.direction);

  // Headlines
  renderNews(d.headlines, d.direction);

  // Show results with animation
  const resultsEl = document.getElementById("results");
  resultsEl.style.display = "flex";
  resultsEl.classList.remove("visible");
  void resultsEl.offsetWidth;          // force reflow
  resultsEl.classList.add("visible");
}

// ── Chart ─────────────────────────────────
function renderChart(chartData, direction) {
  if (priceChart) { priceChart.destroy(); priceChart = null; }

  const ctx    = document.getElementById("priceChart").getContext("2d");
  const labels = chartData.labels;
  const closes = chartData.close;
  const isUp   = direction === "UP";

  // Prediction zone: shade last 5 bars
  const zoneStart = Math.max(0, closes.length - 5);
  const bgColors  = closes.map((_, i) =>
    i >= zoneStart
      ? (isUp ? "rgba(0,229,160,0.12)" : "rgba(255,77,109,0.12)")
      : "transparent"
  );
  const bdColors  = closes.map((_, i) =>
    i >= zoneStart
      ? (isUp ? "rgba(0,229,160,0.5)" : "rgba(255,77,109,0.5)")
      : "transparent"
  );

  // Gradient for price line
  const grad = ctx.createLinearGradient(0, 0, 0, 280);
  grad.addColorStop(0,   "rgba(240,165,0,0.18)");
  grad.addColorStop(1,   "rgba(240,165,0,0)");

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close Price",
          data:  closes,
          borderColor:     "#f0a500",
          borderWidth:     2,
          pointRadius:     0,
          pointHoverRadius:4,
          fill:            true,
          backgroundColor: grad,
          tension:         0.3,
        },
        {
          label:           "Prediction Zone",
          data:            closes.map((v, i) => i >= zoneStart ? v : null),
          borderColor:     isUp ? "#00e5a0" : "#ff4d6d",
          borderWidth:     2,
          borderDash:      [5, 4],
          pointRadius:     0,
          fill:            false,
          tension:         0.3,
          spanGaps:        false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0c1018",
          borderColor:     "rgba(255,255,255,.1)",
          borderWidth:     1,
          titleColor:      "#f0a500",
          bodyColor:       "#dde4ef",
          titleFont:       { family: "'Space Mono'" },
          callbacks: {
            label: ctx => " $" + ctx.parsed.y.toFixed(2),
          },
        },
      },
      scales: {
        x: {
          grid:  { color: "rgba(255,255,255,.04)" },
          ticks: {
            color: "#5a6a82", font: { family: "'Space Mono'", size: 10 },
            maxTicksLimit: 8,
          },
        },
        y: {
          position: "right",
          grid:  { color: "rgba(255,255,255,.04)" },
          ticks: {
            color: "#5a6a82", font: { family: "'Space Mono'", size: 10 },
            callback: v => "$" + v.toFixed(0),
          },
        },
      },
    },
  });
}

// ── News headlines ─────────────────────────
function renderNews(headlines, direction) {
  const list = document.getElementById("newsList");
  list.innerHTML = "";
  const sentimentTag = direction === "UP" ? "Bullish" : "Bearish";

  headlines.forEach((h, i) => {
    const li = document.createElement("li");
    li.className = "news-item";
    li.style.animationDelay = (i * 80) + "ms";
    li.innerHTML = `
      <span class="news-num">#${i + 1}</span>
      <span class="news-text">${h}</span>
      <span class="news-tag">${sentimentTag}</span>
    `;
    list.appendChild(li);
  });
}

// ── Error display ──────────────────────────
function showError(msg) {
  const el = document.getElementById("errorBox");
  el.textContent = "⚠️  " + msg;
  el.style.display = "block";
  hide("spinnerWrap");
}

// ── Enter key support ──────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && document.activeElement.id === "ticker") {
    runPrediction();
  }
});
