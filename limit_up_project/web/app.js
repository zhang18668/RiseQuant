const state = {
  models: [],
  modelId: "",
};

const $ = (id) => document.getElementById(id);

function fmtNum(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${n.toFixed(2)}%`;
}

function hitCell(value) {
  if (value === null || value === undefined) return '<span class="pill pending">待验证</span>';
  return value ? '<span class="pill hit">达到</span>' : '<span class="pill miss">未达</span>';
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function setStatus(text) {
  $("statusText").textContent = text;
}

function renderModelSelect() {
  const select = $("modelSelect");
  select.innerHTML = "";
  const researchModels = state.models.filter((model) => model.has_samples || model.has_candidates);
  const models = researchModels.length ? researchModels : state.models;
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.name} / ${model.version}`;
    select.appendChild(option);
  }
  state.modelId = models[0]?.id || "";
  select.value = state.modelId;
  renderModelMeta(models[0]);
}

function renderModelMeta(model) {
  if (!model) {
    $("modelMeta").innerHTML = "未发现研究结果";
    return;
  }
  const summary = model.summary || {};
  $("modelMeta").innerHTML = `
    <div class="metric"><span>名称</span><strong>${model.name}</strong></div>
    <div class="metric"><span>版本</span><strong>${model.version}</strong></div>
    <div class="metric"><span>初始样本</span><strong>${summary.num_candidates ?? "-"}</strong></div>
    <div class="metric"><span>入选样本</span><strong>${summary.num_samples ?? "-"}</strong></div>
    <div class="metric"><span>触达率</span><strong>${summary.label_3_hit_rate === null || summary.label_3_hit_rate === undefined ? "-" : fmtPct(summary.label_3_hit_rate * 100)}</strong></div>
    <div class="metric"><span>交易流水</span><strong>${model.has_trades ? "已生成" : "缺失"}</strong></div>
  `;
}

function renderDates(dates, selected) {
  const select = $("dateSelect");
  const current = selected || select.value;
  select.innerHTML = "";
  for (const date of dates || []) {
    const option = document.createElement("option");
    option.value = date;
    option.textContent = date;
    select.appendChild(option);
  }
  if (current) select.value = current;
}

function stockLabel(row) {
  return row.name && row.name !== row.code ? `${row.code} ${row.name}` : row.code;
}

function renderSummary(counts) {
  $("shownCount").textContent = counts?.shown ?? 0;
  $("selectedCount").textContent = counts?.selected ?? 0;
  $("candidateCount").textContent = counts?.candidate ?? 0;
  $("hitCount").textContent = counts?.target_hit ?? 0;
}

function renderRows(rows, missing = []) {
  const body = $("researchBody");
  body.innerHTML = "";
  if (missing.length) {
    body.innerHTML = `<tr><td colspan="17" class="empty">缺少 ${missing.join(", ")}，请先运行研究脚本生成结果。</td></tr>`;
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="17" class="empty">当前日期没有符合条件的股票。</td></tr>`;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.className = row.selected_after_factors ? "selected-row" : "";
    tr.innerHTML = `
      <td>${row.date || "-"}</td>
      <td class="stock">${stockLabel(row)}</td>
      <td>${row.selected_after_factors ? '<span class="pill selected">加因子入选</span>' : '<span class="pill base">初始形态</span>'}</td>
      <td>${row.limit_date || "-"}</td>
      <td>${row.wash_date || "-"}</td>
      <td>${fmtNum(row.t_close, 2)}</td>
      <td class="${Number(row.confirm_return_pct) >= 0 ? "up" : "down"}">${fmtPct(row.confirm_return_pct)}</td>
      <td>${fmtPct(row.confirm_high_return_pct)}</td>
      <td>${row.next_date || "-"}</td>
      <td>${hitCell(row.target_hit)}</td>
      <td class="${Number(row.target_high_return_pct) >= 3 ? "up" : ""}">${fmtPct(row.target_high_return_pct)}</td>
      <td class="${Number(row.next_close_return_pct) >= 0 ? "up" : "down"}">${fmtPct(row.next_close_return_pct)}</td>
      <td>${fmtNum(row.sell_price, 2)}</td>
      <td class="${Number(row.pnl_per_share) >= 0 ? "up" : "down"}">${fmtNum(row.pnl_per_share, 3)}</td>
      <td>${fmtPct(row.wash_turnover_rate)}</td>
      <td>${fmtNum(row.volume_ratio, 2)}</td>
      <td>${fmtNum(row.limit_day_vol_ratio_5, 2)}</td>
    `;
    body.appendChild(tr);
  }
}

async function loadResearch() {
  if (!state.modelId) return;
  const date = $("dateSelect").value || "";
  const mode = $("modeSelect").value || "all";
  const top = $("topInput").value || "500";
  const data = await api(`/api/research?model=${encodeURIComponent(state.modelId)}&date=${encodeURIComponent(date)}&mode=${encodeURIComponent(mode)}&top=${encodeURIComponent(top)}`);
  renderModelMeta(data.model);
  renderDates(data.dates, data.selected_date);
  renderSummary(data.counts);
  renderRows(data.rows || [], data.missing || []);
  setStatus(`${data.selected_date || "-"}：展示 ${data.counts?.shown ?? 0} 只，其中加因子入选 ${data.counts?.selected ?? 0} 只，次日触达 3% ${data.counts?.target_hit ?? 0} 只。`);
}

async function init() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    renderModelSelect();
    await loadResearch();
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
    renderRows([], ["本地服务或研究结果"]);
  }

  $("modelSelect").addEventListener("change", async (event) => {
    state.modelId = event.target.value;
    await loadResearch();
  });
  $("dateSelect").addEventListener("change", loadResearch);
  $("modeSelect").addEventListener("change", loadResearch);
  $("refreshBtn").addEventListener("click", loadResearch);
  $("topInput").addEventListener("change", loadResearch);
}

init();
