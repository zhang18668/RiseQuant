const state = {
  models: [],
  modelId: "",
  selectedCode: "",
  selectedName: "",
  chartMode: "daily",
  candles: [],
  markers: [],
  viewCount: 150,
  viewEnd: null,
  dragging: false,
  dragX: 0,
  chartLayout: null,
};

const $ = (id) => document.getElementById(id);

function fmtPct(v, alreadyPercent = false) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  const n = Number(v);
  return `${(alreadyPercent ? n : n * 100).toFixed(2)}%`;
}

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function markerText(action) {
  const a = String(action || "");
  if (a.startsWith("BUY")) return "买入";
  if (a.startsWith("SELL_TAKE_PROFIT")) return "止盈卖出";
  if (a.startsWith("SELL_STOP")) return "止损卖出";
  if (a.startsWith("SELL_TRAILING")) return "移动止盈";
  if (a.startsWith("SELL_TIME")) return "时间卖出";
  if (a.startsWith("SELL")) return "卖出";
  return a || "-";
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function setStatus(text) {
  $("statusText").textContent = text;
}

function renderModelMeta(model) {
  if (!model) {
    $("modelMeta").innerHTML = "未发现模型";
    return;
  }
  const m = model.metrics || {};
  $("modelMeta").innerHTML = `
    <div class="metric"><span>名称</span><strong>${model.name}</strong></div>
    <div class="metric"><span>版本</span><strong>${model.version}</strong></div>
    <div class="metric"><span>Sharpe</span><strong>${fmtNum(m.sharpe_ratio, 2)}</strong></div>
    <div class="metric"><span>总收益</span><strong>${fmtPct(m.total_return)}</strong></div>
    <div class="metric"><span>最大回撤</span><strong>${fmtPct(m.max_drawdown)}</strong></div>
    <div class="metric"><span>交易</span><strong>${m.num_trades ?? "-"}</strong></div>
    <div class="metric"><span>信号文件</span><strong>${model.has_signals ? "已加载" : "缺失"}</strong></div>
    <div class="metric"><span>交易文件</span><strong>${model.has_trades ? "已加载" : "缺失"}</strong></div>
  `;
}

function renderModelSelect() {
  const sel = $("modelSelect");
  sel.innerHTML = "";
  for (const model of state.models) {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = `${model.name} · ${model.version}`;
    sel.appendChild(opt);
  }
  const firstWithSignals = state.models.find((m) => m.has_signals);
  state.modelId = firstWithSignals?.id || state.models[0]?.id || "";
  sel.value = state.modelId;
  renderModelMeta(state.models.find((m) => m.id === state.modelId));
}

function renderDateSelect(dates, selected) {
  const sel = $("dateSelect");
  const current = selected || sel.value;
  sel.innerHTML = "";
  for (const date of dates || []) {
    const opt = document.createElement("option");
    opt.value = date;
    opt.textContent = date;
    sel.appendChild(opt);
  }
  if (current) sel.value = current;
}

function stockLabel(row) {
  return row.name && row.name !== row.code ? `${row.code} ${row.name}` : row.code;
}

function renderSignals(rows, missing = []) {
  const body = $("signalsBody");
  body.innerHTML = "";
  $("signalCount").textContent = `${rows.length} 只`;
  if (missing.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty">缺少 ${missing.join(", ")}，请先生成完整回测产物。</td></tr>`;
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty">当前日期和分数阈值下没有尾盘买入票。宁缺毋滥。</td></tr>`;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rank ?? "-"}</td>
      <td><button class="stock-link" data-code="${row.code}">${stockLabel(row)}</button></td>
      <td>${row.decision_time || "14:30-14:50"}</td>
      <td><span class="badge">${fmtNum(row.score, 3)}</span></td>
      <td><span class="buy">${row.decision || "确认买入"}</span></td>
      <td>${fmtPct(row.position_pct ?? 0.1)}</td>
      <td>${fmtNum(row.close, 2)}</td>
      <td class="${Number(row.change_pct) >= 0 ? "buy" : "sell"}">${fmtPct(row.change_pct, true)}</td>
      <td>${fmtNum(row.stop_loss_price, 2)}</td>
    `;
    body.appendChild(tr);
  }
  body.querySelectorAll(".stock-link").forEach((btn) => {
    btn.addEventListener("click", () => loadKline(btn.dataset.code));
  });
}

async function loadSignals() {
  if (!state.modelId) return;
  const date = $("dateSelect").value;
  const minScore = $("minScoreInput").value || "0";
  const top = $("topInput").value || "10";
  const data = await api(`/api/signals?model=${encodeURIComponent(state.modelId)}&date=${date}&min_score=${minScore}&top=${top}`);
  renderModelMeta(data.model);
  renderDateSelect(data.dates, data.selected_date);
  renderSignals(data.rows, data.missing);
  setStatus(`${data.selected_date} 尾盘 ${data.selection_window || "14:30-14:50"}，确认买入 ${data.rows.length} 只`);
  if (!state.selectedCode && data.rows.length) {
    await loadKline(data.rows[0].code);
  }
}

function renderTrades(rows, missing = []) {
  const body = $("tradesBody");
  body.innerHTML = "";
  if (missing.length) {
    body.innerHTML = `<tr><td colspan="7" class="empty">缺少 ${missing.join(", ")}，无法展示成交明细。</td></tr>`;
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="7" class="empty">没有匹配的成交记录。</td></tr>`;
    return;
  }
  for (const row of rows.slice().reverse()) {
    const action = String(row.action || "");
    const label = row.name && row.name !== row.code ? `${row.code} ${row.name}` : row.code;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.date}</td>
      <td><button class="stock-link" data-code="${row.code}">${label}</button></td>
      <td class="${action.startsWith("BUY") ? "buy" : "sell"}">${markerText(action)}</td>
      <td>${fmtNum(row.price, 2)}</td>
      <td>${fmtNum(row.amount, 0)}</td>
      <td>${fmtNum(row.pnl, 0)}</td>
      <td>${fmtPct(row.return_pct)}</td>
    `;
    body.appendChild(tr);
  }
  body.querySelectorAll(".stock-link").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab("signals");
      loadKline(btn.dataset.code);
    });
  });
}

async function loadTrades() {
  if (!state.modelId) return;
  const code = $("tradeCodeInput").value.trim();
  const data = await api(`/api/trades?model=${encodeURIComponent(state.modelId)}&code=${encodeURIComponent(code)}`);
  renderTrades(data.rows, data.missing);
}

function canvasPrep(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(760, Math.floor(rect.width * ratio));
  canvas.height = Math.max(520, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawEmpty(canvas, text) {
  const { ctx, width, height } = canvasPrep(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#687385";
  ctx.font = "14px Segoe UI, Microsoft YaHei";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
}

function visibleCandles() {
  const total = state.candles.length;
  if (!total) return [];
  if (state.viewEnd === null) state.viewEnd = total;
  state.viewCount = Math.max(35, Math.min(state.viewCount, total));
  state.viewEnd = Math.max(state.viewCount, Math.min(state.viewEnd, total));
  return state.candles.slice(state.viewEnd - state.viewCount, state.viewEnd);
}

function markersForVisible(candles) {
  const dates = new Set(candles.map((d) => d.date));
  return (state.markers || []).filter((m) => dates.has(m.date));
}

function daysBetween(startDate, endDate) {
  if (!startDate || !endDate) return "-";
  const a = new Date(startDate);
  const b = new Date(endDate);
  if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) return "-";
  return `${Math.max(0, Math.round((b - a) / 86400000))}天`;
}

function pairTradeMarkers(markers) {
  const sorted = [...(markers || [])].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const pairs = [];
  let open = null;
  for (const marker of sorted) {
    const action = String(marker.action || "");
    if (action.startsWith("BUY")) {
      if (open) pairs.push({ buy: open, sell: null });
      open = marker;
    } else if (action.startsWith("SELL")) {
      if (open) {
        pairs.push({ buy: open, sell: marker });
        open = null;
      } else {
        pairs.push({ buy: null, sell: marker });
      }
    }
  }
  if (open) pairs.push({ buy: open, sell: null });
  return pairs;
}

function renderTradePath(markers) {
  const body = $("tradePathBody");
  const pairs = pairTradeMarkers(markers);
  body.innerHTML = "";
  $("pathSummary").textContent = pairs.length ? `${pairs.length} 个标记` : "没有买卖标记";
  if (!pairs.length) {
    body.innerHTML = `<tr><td colspan="7" class="empty">该股票暂无尾盘信号或成交记录。</td></tr>`;
    return;
  }
  for (const pair of pairs) {
    const buy = pair.buy || {};
    const sell = pair.sell || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="${buy.date ? "buy" : "trade-open"}">${buy.date || "-"}</td>
      <td>${buy.price ? fmtNum(buy.price, 2) : "-"}</td>
      <td class="${sell.date ? "sell" : "trade-open"}">${sell.date || "未卖出"}</td>
      <td>${sell.price ? fmtNum(sell.price, 2) : "-"}</td>
      <td>${markerText(sell.action) || "-"}</td>
      <td>${daysBetween(buy.date, sell.date)}</td>
      <td class="${Number(sell.return_pct) >= 0 ? "buy" : "sell"}">${sell.date ? fmtPct(sell.return_pct) : "-"}</td>
    `;
    body.appendChild(tr);
  }
}

function strokeLine(ctx, candles, field, color, xFor, yFor) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  let started = false;
  candles.forEach((d, i) => {
    const v = d[field];
    if (v === null || v === undefined || Number.isNaN(Number(v))) return;
    if (!started) {
      ctx.moveTo(xFor(i), yFor(v));
      started = true;
    } else {
      ctx.lineTo(xFor(i), yFor(v));
    }
  });
  if (started) ctx.stroke();
}

function drawKline() {
  const canvas = $("klineCanvas");
  if (state.chartMode === "minute") {
    drawEmpty(canvas, "本地暂未发现通达信分钟数据文件，当前先展示日K。");
    return;
  }
  const candles = visibleCandles();
  if (!candles.length) {
    drawEmpty(canvas, "没有K线数据");
    return;
  }
  const markers = markersForVisible(candles);
  const { ctx, width, height } = canvasPrep(canvas);
  ctx.clearRect(0, 0, width, height);

  const pad = { l: 62, r: 26, t: 28, b: 34 };
  const volH = Math.max(110, height * 0.22);
  const gap = 24;
  const priceH = height - pad.t - pad.b - volH - gap;
  const priceTop = pad.t;
  const volTop = pad.t + priceH + gap;
  const plotW = width - pad.l - pad.r;
  const highs = candles.map((d) => d.high);
  const lows = candles.map((d) => d.low);
  for (const key of ["ma5", "ma10", "ma20"]) {
    for (const d of candles) {
      if (d[key] !== null && d[key] !== undefined) {
        highs.push(d[key]);
        lows.push(d[key]);
      }
    }
  }
  const hi = Math.max(...highs);
  const lo = Math.min(...lows);
  const maxVol = Math.max(...candles.map((d) => d.volume || 0), 1);
  const scaleY = (v) => priceTop + ((hi - v) / Math.max(hi - lo, 0.01)) * priceH;
  const scaleVol = (v) => volTop + volH - (v / maxVol) * volH;
  const step = plotW / candles.length;
  const xFor = (i) => pad.l + i * step + step / 2;
  const bodyW = Math.max(3, Math.min(11, step * 0.62));
  state.chartLayout = { candles, pad, width, height, step, xFor };

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = priceTop + (priceH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    const price = hi - ((hi - lo) / 4) * i;
    ctx.fillStyle = "#687385";
    ctx.font = "12px Segoe UI";
    ctx.textAlign = "right";
    ctx.fillText(price.toFixed(2), pad.l - 8, y + 4);
  }
  ctx.beginPath();
  ctx.moveTo(pad.l, volTop);
  ctx.lineTo(width - pad.r, volTop);
  ctx.stroke();
  ctx.fillStyle = "#687385";
  ctx.fillText("VOL", pad.l - 8, volTop + 14);

  candles.forEach((d, i) => {
    const x = xFor(i);
    const up = d.close >= d.open;
    const color = up ? "#d14b4b" : "#16805b";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, scaleY(d.high));
    ctx.lineTo(x, scaleY(d.low));
    ctx.stroke();
    const yOpen = scaleY(d.open);
    const yClose = scaleY(d.close);
    const top = Math.min(yOpen, yClose);
    const h = Math.max(1, Math.abs(yClose - yOpen));
    if (up) ctx.strokeRect(x - bodyW / 2, top, bodyW, h);
    else ctx.fillRect(x - bodyW / 2, top, bodyW, h);

    const vh = volTop + volH - scaleVol(d.volume || 0);
    ctx.globalAlpha = 0.55;
    ctx.fillRect(x - bodyW / 2, volTop + volH - vh, bodyW, vh);
    ctx.globalAlpha = 1;
  });

  strokeLine(ctx, candles, "ma5", "#f59e0b", xFor, scaleY);
  strokeLine(ctx, candles, "ma10", "#2563eb", xFor, scaleY);
  strokeLine(ctx, candles, "ma20", "#7c3aed", xFor, scaleY);

  ctx.font = "12px Segoe UI, Microsoft YaHei";
  ctx.textAlign = "left";
  ctx.fillStyle = "#f59e0b";
  ctx.fillText("MA5", pad.l + 6, 18);
  ctx.fillStyle = "#2563eb";
  ctx.fillText("MA10", pad.l + 50, 18);
  ctx.fillStyle = "#7c3aed";
  ctx.fillText("MA20", pad.l + 102, 18);

  const markerMap = new Map();
  for (const m of markers) {
    if (!markerMap.has(m.date)) markerMap.set(m.date, []);
    markerMap.get(m.date).push(m);
  }
  candles.forEach((d, i) => {
    const marks = markerMap.get(d.date) || [];
    marks.forEach((m, idx) => {
      const x = xFor(i);
      const action = String(m.action || "");
      const isBuy = action.startsWith("BUY");
      const y = scaleY(m.price || d.close);
      const color = isBuy ? "#16805b" : "#d14b4b";
      ctx.save();
      ctx.strokeStyle = color.replace(")", ", 0.35)");
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, priceTop);
      ctx.lineTo(x, volTop + volH);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = color;
      ctx.beginPath();
      if (isBuy) {
        ctx.moveTo(x, y - 14 - idx * 13);
        ctx.lineTo(x - 7, y - 3 - idx * 13);
        ctx.lineTo(x + 7, y - 3 - idx * 13);
      } else {
        ctx.moveTo(x, y + 14 + idx * 13);
        ctx.lineTo(x - 7, y + 3 + idx * 13);
        ctx.lineTo(x + 7, y + 3 + idx * 13);
      }
      ctx.closePath();
      ctx.fill();

      const reason = isBuy ? fmtNum(m.price, 2) : `${markerText(action)} ${fmtPct(m.return_pct)}`;
      const label = `${isBuy ? "买" : "卖"} ${String(m.date).slice(5)} ${reason}`;
      const labelX = Math.min(width - pad.r - 112, Math.max(pad.l + 4, x + 8));
      const labelY = Math.min(volTop - 10, Math.max(priceTop + 16, y + (isBuy ? -20 : 26) + idx * 14));
      const textW = ctx.measureText(label).width + 10;
      ctx.fillStyle = color;
      ctx.fillRect(labelX, labelY - 12, textW, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, labelX + 5, labelY + 2);
    });
  });

  ctx.fillStyle = "#687385";
  ctx.textAlign = "center";
  ctx.font = "12px Segoe UI";
  const ticks = [0, Math.floor(candles.length / 3), Math.floor((candles.length * 2) / 3), candles.length - 1];
  ticks.forEach((idx) => {
    const x = xFor(idx);
    ctx.fillText(candles[idx].date.slice(5), x, height - 12);
  });
}

async function loadKline(code) {
  if (!code) return;
  state.selectedCode = code;
  const data = await api(`/api/kline?model=${encodeURIComponent(state.modelId)}&code=${encodeURIComponent(code)}`);
  state.candles = data.candles || [];
  state.markers = data.markers || [];
  state.selectedName = data.name || code;
  state.viewCount = Math.min(150, Math.max(35, state.candles.length));
  state.viewEnd = state.candles.length;
  if (state.markers.length && state.candles.length) {
    const markerDate = state.markers[state.markers.length - 1].date;
    const idx = state.candles.findIndex((d) => d.date >= markerDate);
    if (idx >= 0) state.viewEnd = Math.min(state.candles.length, idx + 45);
  }
  $("chartTitle").textContent = `${code} ${state.selectedName} · K线与买卖结构`;
  $("chartHint").textContent = "移动鼠标查看日期、开高低收、成交量、成交额、换手率；绿色为确认买入点，红色为确认卖出点。";
  drawKline();
  renderTradePath(state.markers);
}

function drawEquity(rows) {
  const canvas = $("equityCanvas");
  if (!rows.length) {
    drawEmpty(canvas, "没有净值曲线");
    return;
  }
  const { ctx, width, height } = canvasPrep(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { l: 66, r: 24, t: 22, b: 40 };
  const vals = rows.map((r) => r.total_value);
  const hi = Math.max(...vals);
  const lo = Math.min(...vals);
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const x = (i) => pad.l + (i / Math.max(rows.length - 1, 1)) * plotW;
  const y = (v) => pad.t + ((hi - v) / Math.max(hi - lo, 1)) * plotH;
  ctx.strokeStyle = "#e5e7eb";
  for (let i = 0; i <= 4; i += 1) {
    const yy = pad.t + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(width - pad.r, yy);
    ctx.stroke();
  }
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((r, i) => {
    if (i === 0) ctx.moveTo(x(i), y(r.total_value));
    else ctx.lineTo(x(i), y(r.total_value));
  });
  ctx.stroke();
  $("equitySummary").textContent = `最终 ${fmtNum(vals[vals.length - 1], 0)} / 最高 ${fmtNum(hi, 0)}`;
}

async function loadEquity() {
  if (!state.modelId) return;
  const data = await api(`/api/equity?model=${encodeURIComponent(state.modelId)}`);
  drawEquity(data.rows || []);
}

async function showMinute() {
  state.chartMode = "minute";
  $("dailyBtn").classList.remove("active");
  $("minuteBtn").classList.add("active");
  if (!state.selectedCode) return;
  const data = await api(`/api/minute?model=${encodeURIComponent(state.modelId)}&code=${encodeURIComponent(state.selectedCode)}`);
  drawEmpty($("klineCanvas"), (data.missing || ["暂无分时数据"])[0]);
}

function showDaily() {
  state.chartMode = "daily";
  $("minuteBtn").classList.remove("active");
  $("dailyBtn").classList.add("active");
  drawKline();
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(`${name}View`).classList.add("active");
  $("pageTitle").textContent = name === "signals" ? "14:30-14:50 尾盘确认买入" : name === "trades" ? "交易复盘" : "净值曲线";
  if (name === "trades") loadTrades();
  if (name === "equity") loadEquity();
}

function showTooltip(event) {
  const tip = $("klineTooltip");
  if (!state.chartLayout || state.chartMode !== "daily") return;
  const canvas = $("klineCanvas");
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const { candles, pad, width, step } = state.chartLayout;
  if (x < pad.l || x > width - pad.r) {
    tip.style.display = "none";
    return;
  }
  const idx = Math.max(0, Math.min(candles.length - 1, Math.floor((x - pad.l) / step)));
  const d = candles[idx];
  const marks = (state.markers || []).filter((m) => m.date === d.date).map((m) => markerText(m.action)).join(" / ");
  tip.innerHTML = `
    <strong>${d.date}</strong>${marks ? ` · ${marks}` : ""}<br>
    开 ${fmtNum(d.open, 2)}　高 ${fmtNum(d.high, 2)}　低 ${fmtNum(d.low, 2)}　收 ${fmtNum(d.close, 2)}<br>
    涨跌幅 ${fmtPct(d.change_pct, true)}　成交量 ${fmtNum(d.volume, 0)}　成交额 ${fmtNum(d.amount, 0)}<br>
    换手率 ${d.turnover_rate === null || d.turnover_rate === undefined ? "暂无流通股本数据" : fmtPct(d.turnover_rate, true)}
  `;
  tip.style.display = "block";
  tip.style.left = `${Math.min(rect.width - 360, Math.max(8, event.clientX - rect.left + 14))}px`;
  tip.style.top = `${Math.max(8, event.clientY - rect.top - 94)}px`;
}

function wireChartInteractions() {
  const canvas = $("klineCanvas");
  canvas.addEventListener("wheel", (event) => {
    if (!state.candles.length || state.chartMode !== "daily") return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1.14 : 0.86;
    state.viewCount = Math.round(state.viewCount * factor);
    state.viewCount = Math.max(35, Math.min(state.viewCount, state.candles.length));
    state.viewEnd = Math.max(state.viewCount, Math.min(state.viewEnd ?? state.candles.length, state.candles.length));
    drawKline();
  }, { passive: false });
  canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.dragX = event.clientX;
  });
  canvas.addEventListener("mousemove", showTooltip);
  canvas.addEventListener("mouseleave", () => {
    $("klineTooltip").style.display = "none";
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
  });
  window.addEventListener("mousemove", (event) => {
    if (!state.dragging || !state.candles.length || state.chartMode !== "daily") return;
    const dx = event.clientX - state.dragX;
    if (Math.abs(dx) < 8) return;
    const bars = Math.round(dx / 8);
    state.viewEnd = Math.max(state.viewCount, Math.min((state.viewEnd ?? state.candles.length) - bars, state.candles.length));
    state.dragX = event.clientX;
    drawKline();
  });
}

async function init() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    renderModelSelect();
    await loadSignals();
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
    drawEmpty($("klineCanvas"), "服务数据加载失败");
  }

  $("collapseBtn").addEventListener("click", () => document.body.classList.toggle("sidebar-collapsed"));
  $("modelSelect").addEventListener("change", async (event) => {
    state.modelId = event.target.value;
    state.selectedCode = "";
    renderModelMeta(state.models.find((m) => m.id === state.modelId));
    await loadSignals();
  });
  $("refreshBtn").addEventListener("click", loadSignals);
  $("dateSelect").addEventListener("change", loadSignals);
  $("zoomInBtn").addEventListener("click", () => {
    state.viewCount = Math.max(35, Math.round(state.viewCount * 0.82));
    drawKline();
  });
  $("zoomOutBtn").addEventListener("click", () => {
    state.viewCount = Math.min(state.candles.length || 240, Math.round(state.viewCount * 1.2));
    drawKline();
  });
  $("resetZoomBtn").addEventListener("click", () => {
    state.viewCount = Math.min(150, state.candles.length || 150);
    state.viewEnd = state.candles.length;
    drawKline();
  });
  $("dailyBtn").addEventListener("click", showDaily);
  $("minuteBtn").addEventListener("click", showMinute);
  $("tradeCodeInput").addEventListener("input", () => {
    clearTimeout(window.__tradeTimer);
    window.__tradeTimer = setTimeout(loadTrades, 180);
  });
  document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
  window.addEventListener("resize", () => {
    if (state.selectedCode) drawKline();
  });
  wireChartInteractions();
}

init();
