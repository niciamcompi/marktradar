"use strict";
// Marktradar web UI. Deliberately framework-free and without CDNs: works offline,
// inside corporate networks, and stays readable in one file.

const SVG = "http://www.w3.org/2000/svg";
const nf = (d = 0) => new Intl.NumberFormat("de-CH", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtInt = (v) => nf(0).format(Math.round(v));
const fmtSigned = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + nf(d).format(Math.abs(v));
const fmtCompact = (v) => {
  const a = Math.abs(v);
  const s = a >= 1000 ? nf(a >= 10000 ? 0 : 1).format(a / 1000) + "k" : nf(0).format(a);
  return (v < 0 ? "−" : v > 0 ? "+" : "") + s;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = (id) => document.getElementById(id);

const state = {
  meta: null, versicherer: [], horizon: 2, qualityHorizon: 2, kanton: "", filter: "gross",
  markt: [], insurer: null, curve: [], actual: null, scenarioPp: null,
};

async function api(path, body) {
  const res = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function el(tag, attrs = {}, parent) {
  const node = tag.startsWith("svg:") ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function table(head, rows) {
  const th = head.map((h, i) => `<th class="${i ? "n" : ""}">${h}</th>`).join("");
  const tb = rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "n" : ""}">${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`;
}

function segmented(host, options, current, onChange) {
  host.innerHTML = "";
  for (const [value, label] of options) {
    const b = el("button", { type: "button", "aria-pressed": String(value === current), text: label }, host);
    b.addEventListener("click", () => {
      host.querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      onChange(value);
    });
  }
}

const yearOf = (h) => state.meta.basisjahr + h;

// ---------- Tooltip ----------
const tip = $("tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, window.innerWidth - r.width - 8)}px`;
  tip.style.top = `${y + r.height + 20 > window.innerHeight ? y - r.height - 12 : y + 14}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---------- Dot plot (used for the market and the canton breakdown) ----------
// rows: [{label, value, lo, hi, tooltip, onClick?}]
function dotplot(host, rows, valueHeader, labelHeader = "Versicherer") {
  host.innerHTML = "";
  if (!rows.length) { host.innerHTML = `<p class="empty">Keine Daten.</p>`; return; }
  const narrow = window.innerWidth <= 600;
  const lo = Math.min(-1, ...rows.map((r) => r.lo));
  const hi = Math.max(1, ...rows.map((r) => r.hi));
  const plotW = host.getBoundingClientRect().width - (narrow ? 176 : 302);
  const ticks = niceTicks(lo, hi, plotW < 260 ? 3 : 6);
  const domain = [Math.min(lo, ticks[0]), Math.max(hi, ticks[ticks.length - 1])];

  const axis = el("div", { class: "dp-axis" }, host);
  el("span", { text: labelHeader }, axis);
  const axisCell = el("div", {}, axis);
  el("span", { class: "dp-val", text: valueHeader }, axis);

  const cells = rows.map((r) => {
    const row = el("div", { class: `dp-row${r.onClick ? "" : " static"}` }, host);
    const name = el("button", { class: `dp-name${r.onClick ? "" : " static"}`, type: "button", text: r.label, title: r.label }, row);
    if (r.onClick) name.addEventListener("click", r.onClick);
    const cell = el("div", {}, row);
    el("span", { class: "dp-val", text: `${fmtSigned(r.value)} %` }, row);
    row.addEventListener("mousemove", (e) => showTip(r.tooltip, e.clientX, e.clientY));
    row.addEventListener("mouseleave", hideTip);
    return cell;
  });

  const width = axisCell.getBoundingClientRect().width || 400;
  const pad = 8;
  const x = (v) => pad + ((v - domain[0]) / (domain[1] - domain[0])) * (width - 2 * pad);
  const axSvg = el("svg:svg", { width, height: 26, class: "axis" }, axisCell);
  for (const t of ticks) el("svg:text", { x: x(t), y: 17, "text-anchor": "middle", text: t === 0 ? "0" : `${fmtSigned(t, 0)} %` }, axSvg);

  rows.forEach((r, i) => {
    const s = el("svg:svg", { width, height: 26 }, cells[i]);
    for (const t of ticks) el("svg:line", { x1: x(t), x2: x(t), y1: 0, y2: 26, class: t === 0 ? "zeroline" : "gridline" }, s);
    el("svg:line", { x1: x(r.lo), x2: x(r.hi), y1: 13, y2: 13, stroke: css("--range"), "stroke-width": 2, "stroke-linecap": "round" }, s);
    el("svg:circle", { cx: x(r.value), cy: 13, r: 5, fill: r.value >= 0 ? css("--gain") : css("--loss"), stroke: css("--paper"), "stroke-width": 2 }, s);
  });
}

// ---------- 01 Markt ----------
async function loadMarkt() {
  const q = new URLSearchParams({ horizont: state.horizon });
  if (state.kanton) q.set("kanton", state.kanton);
  state.markt = await api(`/markt?${q}`);
  renderMarkt();
}

function renderMarkt() {
  const minSize = state.kanton ? 10000 : 100000;
  const rows = state.filter === "gross" ? state.markt.filter((r) => r.bestand_basis >= minSize) : state.markt;
  const where = state.kanton ? `im Kanton ${state.kanton}` : "in der Schweiz";
  $("markt-lede").textContent =
    `Erwartete Veränderung des Marktanteils ${where} von ${state.meta.basisjahr} bis ${yearOf(state.horizon)}, ` +
    `auf Basis der vom BAG genehmigten Prämien.`;
  document.querySelector('[data-filter="gross"]').textContent = `≥ ${fmtInt(minSize)} Versicherte`;
  dotplot($("markt-chart"), rows.map((r) => ({
    label: r.name, value: r.anteilswachstum_pct, lo: r.anteilswachstum_untergrenze_pct, hi: r.anteilswachstum_obergrenze_pct,
    onClick: () => selectInsurer(r.bag_nr, true),
    tooltip: `<b>${r.name}</b><br>Marktanteil ${nf(2).format(r.marktanteil_basis_pct)} % (${r.basisjahr})<br>` +
      `Prämie ${r.prognosejahr} vs. Markt: ${fmtSigned(r.praemienanpassung_tatsaechlich_pp)} %-Punkte<br>` +
      `Prognose ${r.prognosejahr}: <b>${fmtSigned(r.anteilswachstum_pct)} %</b><br>` +
      `<span class="muted">80 %: ${fmtSigned(r.anteilswachstum_untergrenze_pct)} bis ${fmtSigned(r.anteilswachstum_obergrenze_pct)} %</span><br>` +
      `Bestand ${fmtInt(r.bestand_basis)} → ${fmtInt(r.bestand_prognose)}`,
  })), "Prognose");
  $("markt-table").innerHTML = table(
    ["Versicherer", "Marktanteil %", "Prämie vs. Markt (pp)", "Prognose %", "Untergrenze %", "Obergrenze %", "Bestand", "Bestand Prognose"],
    rows.map((r) => [r.name, nf(2).format(r.marktanteil_basis_pct), fmtSigned(r.praemienanpassung_tatsaechlich_pp),
      fmtSigned(r.anteilswachstum_pct), fmtSigned(r.anteilswachstum_untergrenze_pct),
      fmtSigned(r.anteilswachstum_obergrenze_pct), fmtInt(r.bestand_basis), fmtInt(r.bestand_prognose)]));
}

// ---------- 02 Szenario ----------
async function selectInsurer(bagNr, scroll = false) {
  state.insurer = bagNr;
  state.scenarioPp = null;
  $("insurer").value = String(bagNr);
  if (scroll) $("szenario").scrollIntoView();
  const h = state.horizon;
  [state.curve, state.actual] = await Promise.all([
    api("/elastizitaet", { bag_nr: bagNr, horizont: h, von_pp: -5, bis_pp: 5, schritt_pp: 0.5 }),
    api("/prognose", { bag_nr: bagNr, horizont: h }),
  ]);
  $("pp").value = String(Math.max(-5, Math.min(5, Math.round(state.actual.praemienanpassung_tatsaechlich_pp * 2) / 2)));
  renderScenario(state.actual);
}

let pending;
function onSlider() {
  state.scenarioPp = +$("pp").value;
  const point = state.curve.find((p) => Math.abs(p.praemienanpassung_rel_pp - state.scenarioPp) < 1e-6);
  if (point) renderScenario({ ...point, kantone: null });
  clearTimeout(pending);
  pending = setTimeout(async () => {
    const res = await api("/prognose", { bag_nr: state.insurer, horizont: state.horizon, praemienanpassung_rel_pp: state.scenarioPp });
    if (state.scenarioPp === res.praemienanpassung_rel_pp) renderScenario(res);
  }, 150);
}

function renderScenario(p) {
  if (p.kantone) state.lastScenario = p;
  const actualPp = state.actual.praemienanpassung_tatsaechlich_pp;
  const pp = p.praemienanpassung_rel_pp ?? actualPp;
  const isActual = p.praemienanpassung_rel_pp === null;
  $("pp-label").textContent = `Prämienanpassung ${p.prognosejahr} relativ zum Markt`;
  $("pp-out").textContent = `${fmtSigned(pp, 1)} %-Punkte${isActual ? " (genehmigt)" : ""}`;
  $("pp-reset").hidden = isActual;
  const delta = p.bestand_prognose - p.bestand_basis;
  $("figures").innerHTML = [
    figure(`Marktanteil ${p.basisjahr}`, `${nf(2).format(p.marktanteil_basis_pct)} %`, `${fmtInt(p.bestand_basis)} Versicherte`),
    figure(`Veränderung Marktanteil bis ${p.prognosejahr}`, `${fmtSigned(p.anteilswachstum_pct)} %`,
      `80 %-Intervall ${fmtSigned(p.anteilswachstum_untergrenze_pct)} bis ${fmtSigned(p.anteilswachstum_obergrenze_pct)} %`),
    figure(`Bestand ${p.prognosejahr}`, fmtInt(p.bestand_prognose), `${fmtSigned(delta, 0)} Personen, bei konstanter Marktgrösse`),
  ].join("");
  renderCurve(pp, actualPp, delta);
  if (p.kantone) renderCantons(p);
}

function figure(label, value, sub) {
  return `<div class="figure"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function renderCantons(p) {
  $("kantone-caption").textContent =
    `Nach Kanton: Veränderung des Marktanteils bis ${p.prognosejahr} (Kantone mit mindestens 1'000 Versicherten)`;
  const rows = p.kantone.filter((k) => k.versicherte >= 1000);
  dotplot($("kantone-chart"), rows.map((k) => ({
    label: `${k.kanton} · ${fmtInt(k.versicherte)}`, value: k.anteilswachstum_pct, lo: k.untergrenze_pct, hi: k.obergrenze_pct,
    tooltip: `<b>${p.name}, ${k.kanton}</b><br>Marktanteil im Kanton ${nf(1).format(k.marktanteil_pct)} %<br>` +
      `Prämie ${p.prognosejahr} vs. Kantonsmarkt: ${fmtSigned(k.praemienanpassung_pp)} %-Punkte<br>` +
      `Prognose: <b>${fmtSigned(k.anteilswachstum_pct)} %</b> (${fmtSigned(k.bestand_veraenderung, 0)} Personen)<br>` +
      `<span class="muted">80 %: ${fmtSigned(k.untergrenze_pct)} bis ${fmtSigned(k.obergrenze_pct)} %</span>`,
  })), "Prognose", "Kanton · Versicherte");
}

function renderCurve(selectedPp, actualPp, selectedDelta) {
  const host = $("curve");
  host.innerHTML = "";
  const W = host.getBoundingClientRect().width || 600;
  const H = 300, m = { t: 22, r: 16, b: 34, l: 56 };
  const pts = state.curve.map((p) => {
    const toPersons = (pct) => p.bestand_basis * pct / 100;
    return { pp: p.praemienanpassung_rel_pp, y: p.bestand_prognose - p.bestand_basis,
      lo: toPersons(p.anteilswachstum_untergrenze_pct), hi: toPersons(p.anteilswachstum_obergrenze_pct), p };
  });
  const yMin = Math.min(0, ...pts.map((d) => d.lo)), yMax = Math.max(0, ...pts.map((d) => d.hi));
  const yt = niceTicks(yMin, yMax, 5);
  const y0 = Math.min(yMin, yt[0]), y1 = Math.max(yMax, yt[yt.length - 1]);
  const x = (v) => m.l + ((v + 5) / 10) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const s = el("svg:svg", { width: W, height: H, role: "img", "aria-label": "Veränderung des Bestands nach Prämienanpassung" }, host);
  const ax = el("svg:g", { class: "axis" }, s);
  for (const t of yt) {
    el("svg:line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? "zeroline" : "gridline" }, ax);
    el("svg:text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: fmtCompact(t) }, ax);
  }
  for (let t = -5; t <= 5; t += 1) el("svg:text", { x: x(t), y: H - 12, "text-anchor": "middle", text: t === 0 ? "0" : fmtSigned(t, 0) }, ax);
  el("svg:text", { x: W - m.r, y: H, "text-anchor": "end", text: "%-Punkte vs. Markt" }, ax);

  const area = pts.map((d) => `${x(d.pp)},${y(d.hi)}`).concat(pts.slice().reverse().map((d) => `${x(d.pp)},${y(d.lo)}`));
  el("svg:polygon", { points: area.join(" "), fill: css("--band") }, s);
  el("svg:polyline", { points: pts.map((d) => `${x(d.pp)},${y(d.y)}`).join(" "), fill: "none",
    stroke: css("--series-1"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);

  // approved tariff as a reference line
  if (actualPp >= -5 && actualPp <= 5) {
    el("svg:line", { x1: x(actualPp), x2: x(actualPp), y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1 }, s);
    el("svg:text", { x: x(actualPp), y: m.t - 8, "text-anchor": "middle", class: "direct-label", text: `genehmigt ${fmtSigned(actualPp)}` }, s);
  }
  // the selected point is drawn exactly where the figures above say it is
  const sel = { pp: selectedPp, y: selectedDelta };
  el("svg:circle", { cx: x(sel.pp), cy: y(sel.y), r: 6, fill: css("--series-1"), stroke: css("--paper"), "stroke-width": 2 }, s);
  const right = x(sel.pp) > W * 0.7, below = sel.y < 0;
  el("svg:text", { x: x(sel.pp) + (right ? -12 : 12), y: y(sel.y) + (below ? 20 : -12), "text-anchor": right ? "end" : "start",
    class: "direct-label", text: `${fmtSigned(sel.y, 0)} Personen` }, s);

  const cross = el("svg:line", { y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, s);
  const hit = el("svg:rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, s);
  const ppAt = (e) => ((e.clientX - s.getBoundingClientRect().left - m.l) / (W - m.l - m.r)) * 10 - 5;
  hit.addEventListener("mousemove", (e) => {
    const pp = ppAt(e);
    const d = pts.reduce((a, b) => (Math.abs(b.pp - pp) < Math.abs(a.pp - pp) ? b : a));
    cross.setAttribute("x1", x(d.pp)); cross.setAttribute("x2", x(d.pp)); cross.setAttribute("visibility", "visible");
    showTip(`<b>${fmtSigned(d.pp, 1)} %-Punkte</b> vs. Markt<br>Marktanteil ${fmtSigned(d.p.anteilswachstum_pct)} %<br>` +
      `Bestand ${fmtSigned(d.y, 0)} Personen<br><span class="muted">80 %: ${fmtCompact(d.lo)} bis ${fmtCompact(d.hi)}</span>`,
      e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("click", (e) => {
    $("pp").value = String(Math.max(-5, Math.min(5, Math.round(ppAt(e) * 2) / 2)));
    onSlider();
  });
}

// ---------- 03 Modellgüte ----------
function renderQuality() {
  const meta = state.meta;
  const hm = meta.horizonte[String(state.qualityHorizon)];
  const b = hm.backtest;
  const rows = hm.backtest_pro_jahr;
  const ahead = state.qualityHorizon === 1 ? "ein Jahr" : "zwei Jahre";
  document.querySelector("#guete .lede").textContent =
    `Für jedes vergangene Jahr wurde das Modell nur mit den Daten trainiert, die damals schon bekannt waren, ` +
    `und dann an der Realität gemessen, so als hätte man es damals eingesetzt. Hier: Prognosen ${ahead} im Voraus.`;
  $("quality-figures").innerHTML = [
    figure("Genauer als die einfache Annahme", `${nf(0).format(b.verbesserung_vs_konstant * 100)} %`,
      `kleinerer Fehler als „Marktanteil bleibt gleich“, über ${b.testjahre} Testjahre`),
    figure("Gewinner und Verlierer", nf(2).format(b.rangkorrelation),
      "Rangkorrelation: 1 = Reihenfolge perfekt, 0 = Zufall"),
    figure("Unsicherheit ehrlich angegeben", `${nf(0).format(b.intervall_abdeckung * 100)} %`,
      `der echten Werte im 80 %-Intervall (vor Kalibrierung ${nf(0).format(b.intervall_abdeckung_roh * 100)} %)`),
  ].join("");
  renderErrorChart(rows);
  $("guete-table").innerHTML = table(
    ["Prognosejahr", "Fehler Modell %", "Fehler „bleibt gleich“ %", "Rangkorrelation", "Intervall-Treffer %", "Versicherer"],
    rows.map((r) => [r.prognosejahr, nf(1).format(r.mae_modell * 100), nf(1).format(r.mae_konstant * 100),
      nf(2).format(r.rangkorrelation), nf(0).format(r.intervall_abdeckung * 100), r.n_versicherer]));
  renderLive(meta.live_monitoring || []);
  renderComparison();
  const facts = [
    ["Modellversion", meta.version],
    ["Datenstand", `Versicherte bis ${meta.basisjahr}, genehmigte Prämien bis ${yearOf(Math.max(...meta.horizonte_verfuegbar))}`],
    ["Training", `${fmtInt(hm.n_trainingszeilen)} Versicherer×Kanton×Jahr, Basisjahre bis ${hm.trainiert_bis_basisjahr}`],
    ["Datenquelle", `<a href="https://opendata.swiss/de/dataset/health-insurance-premiums">BAG Prämien-Open-Data</a>`],
    ["Daten-Fingerabdruck", `<code>${(meta.daten_fingerprint || "–").slice(0, 12)}</code>`],
    ["Code-Stand", meta.git_commit ? `<code>${meta.git_commit}</code>` : "–"],
    ["Modell", "Gradient Boosting je Kanton, monotone Prämien-Constraints, Conformal-Intervalle"],
  ];
  $("facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderErrorChart(rows) {
  const host = $("error-chart");
  host.innerHTML = "";
  const W = host.getBoundingClientRect().width || 600;
  const H = 260, m = { t: 14, r: W < 520 ? 96 : 130, b: 30, l: 40 };
  const series = [
    { key: "mae_modell", name: "Modell", color: css("--series-1") },
    { key: "mae_konstant", name: "bleibt gleich", color: css("--series-2") },
  ];
  const years = rows.map((r) => r.prognosejahr);
  const vmax = Math.max(...rows.flatMap((r) => series.map((s) => r[s.key] * 100)));
  const yt = niceTicks(0, vmax, 4);
  const top = Math.max(vmax, yt[yt.length - 1]);
  const x = (yr) => m.l + ((yr - years[0]) / Math.max(1, years[years.length - 1] - years[0])) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - v / top) * (H - m.t - m.b);

  const s = el("svg:svg", { width: W, height: H, role: "img", "aria-label": "Prognosefehler je Jahr" }, host);
  const ax = el("svg:g", { class: "axis" }, s);
  for (const t of yt) {
    el("svg:line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? "zeroline" : "gridline" }, ax);
    el("svg:text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: `${nf(0).format(t)} %` }, ax);
  }
  const step = W < 520 ? 2 : 1;
  years.forEach((yr, i) => {
    if (i % step === 0 || i === years.length - 1) el("svg:text", { x: x(yr), y: H - 8, "text-anchor": "middle", text: yr }, ax);
  });
  for (const se of series) {
    el("svg:polyline", { points: rows.map((r) => `${x(r.prognosejahr)},${y(r[se.key] * 100)}`).join(" "),
      fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);
    for (const r of rows) el("svg:circle", { cx: x(r.prognosejahr), cy: y(r[se.key] * 100), r: 4, fill: se.color, stroke: css("--paper"), "stroke-width": 2 }, s);
  }
  // direct end labels; nudge apart if they would overlap
  const last = rows[rows.length - 1];
  const labels = series.map((se) => ({ se, y: y(last[se.key] * 100) + 4 })).sort((a, b) => a.y - b.y);
  if (labels[1].y - labels[0].y < 14) labels[1].y = labels[0].y + 14;
  for (const { se, y: ly } of labels) el("svg:text", { x: W - m.r + 10, y: ly, class: "direct-label", text: `${se.name} ${nf(1).format(last[se.key] * 100)} %` }, s);

  const cross = el("svg:line", { y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, s);
  const hit = el("svg:rect", { x: m.l - 10, y: m.t, width: W - m.l - m.r + 20, height: H - m.t - m.b, fill: "transparent" }, s);
  hit.addEventListener("mousemove", (e) => {
    const px = e.clientX - s.getBoundingClientRect().left;
    const r = rows.reduce((a, b) => (Math.abs(x(b.prognosejahr) - px) < Math.abs(x(a.prognosejahr) - px) ? b : a));
    cross.setAttribute("x1", x(r.prognosejahr)); cross.setAttribute("x2", x(r.prognosejahr)); cross.setAttribute("visibility", "visible");
    showTip(`<b>Prognose für ${r.prognosejahr}</b> (erstellt ${r.basisjahr})<br>Modell: ${nf(1).format(r.mae_modell * 100)} %<br>` +
      `„Bleibt gleich“: ${nf(1).format(r.mae_konstant * 100)} %<br><span class="muted">Rangkorrelation ${nf(2).format(r.rangkorrelation)} · ${r.n_versicherer} Versicherer</span>`,
      e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

function renderComparison() {
  const all = (state.vergleich?.zusammenfassung || []).filter((r) => r.horizont === state.qualityHorizon);
  $("vergleich").hidden = !all.length;
  if (!all.length) return;
  const rows = all.filter((r) => r.modell !== "konstant");
  const host = $("vergleich-chart");
  host.innerHTML = "";
  const vals = rows.map((r) => r.verbesserung_vs_konstant * 100);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const cells = rows.map((r) => {
    const row = el("div", { class: "bl-row" }, host);
    const name = el("div", { class: "bl-name" }, row);
    name.innerHTML = `${r.modell === "gbm" ? "<b>" : ""}${r.name}${r.modell === "gbm" ? "</b>" : ""}` +
      `<small>${r.familie}${r.jahr_methode ? ` · ${r.jahr_methode}` : ""}</small>`;
    const cell = el("div", {}, row);
    el("span", { class: "bl-val", text: `${fmtSigned(r.verbesserung_vs_konstant * 100, 0)} %` }, row);
    row.addEventListener("mousemove", (e) => showTip(
      `<b>${r.name}</b><br>Fehler ${nf(1).format(r.mae_modell * 100)} % (Baseline ${nf(1).format(r.mae_konstant * 100)} %)<br>` +
      `Rangkorrelation ${r.rangkorrelation == null ? "–" : nf(2).format(r.rangkorrelation)}<br>` +
      `<span class="muted">${nf(1).format(r.sekunden_pro_jahr)} s Rechenzeit pro Testjahr</span>`, e.clientX, e.clientY));
    row.addEventListener("mouseleave", hideTip);
    return cell;
  });
  const width = cells[0].getBoundingClientRect().width || 300;
  const pad = 6, x = (v) => pad + ((v - lo) / (hi - lo || 1)) * (width - 2 * pad);
  rows.forEach((r, i) => {
    const v = vals[i];
    const s = el("svg:svg", { width, height: 34 }, cells[i]);
    el("svg:line", { x1: x(0), x2: x(0), y1: 0, y2: 34, class: "zeroline" }, s);
    const x0 = Math.min(x(0), x(v)), w = Math.max(1, Math.abs(x(v) - x(0)));
    // bar with a rounded data end, square at the zero baseline
    const rad = Math.min(4, w / 2), y0 = 9, bh = 16, right = v >= 0;
    const d = right
      ? `M${x0},${y0} h${w - rad} a${rad},${rad} 0 0 1 ${rad},${rad} v${bh - 2 * rad} a${rad},${rad} 0 0 1 ${-rad},${rad} h${-(w - rad)} z`
      : `M${x0 + w},${y0} h${-(w - rad)} a${rad},${rad} 0 0 0 ${-rad},${rad} v${bh - 2 * rad} a${rad},${rad} 0 0 0 ${rad},${rad} h${w - rad} z`;
    el("svg:path", { d, fill: v >= 0 ? css("--gain") : css("--loss") }, s);
  });
  const ahead = state.qualityHorizon === 1 ? "ein Jahr" : "zwei Jahre";
  $("vergleich-caption").textContent =
    `Wie viel kleiner ist der Fehler als bei „Marktanteil bleibt gleich“ (Prognosen ${ahead} im Voraus)? ` +
    `Alle Modelle im gleichen Backtest; fett = Modell im Einsatz.`;
  $("vergleich-table").innerHTML = table(
    ["Modell", "Familie", "Fehler %", "Besser als Baseline", "Rangkorrelation", "80 %-Abdeckung (roh)", "Sekunden/Jahr"],
    all.map((r) => [r.name, r.familie, nf(1).format(r.mae_modell * 100), `${fmtSigned(r.verbesserung_vs_konstant * 100, 0)} %`,
      r.rangkorrelation == null ? "–" : nf(2).format(r.rangkorrelation),
      r.intervall_abdeckung == null ? "–" : `${nf(0).format(r.intervall_abdeckung * 100)} %`, nf(1).format(r.sekunden_pro_jahr)]));
}

function renderLive(records) {
  const host = $("live");
  if (!records.length) {
    host.innerHTML = `<p class="empty">Noch keine Live-Prüfung. Sobald das BAG neue Daten publiziert (jeweils Ende September), ` +
      `misst <code>marktradar update</code> das bisherige Modell an Jahren, die es nie gesehen hat, und trainiert danach neu. ` +
      `Die Resultate erscheinen hier.</p>`;
    return;
  }
  host.innerHTML = table(["Prognosejahr", "Horizont", "Modellversion", "Fehler Modell %", "Fehler „bleibt gleich“ %", "Rangkorr."],
    records.map((r) => [r.prognosejahr, `${r.horizont} J.`, `<code>${r.modell_version}</code>`, nf(1).format(r.mae_modell * 100),
      nf(1).format(r.mae_konstant * 100), nf(2).format(r.rangkorrelation)]));
}

// ---------- Init ----------
function wire() {
  document.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    renderMarkt();
  }));
  document.querySelectorAll("[data-toggle-table]").forEach((b) => b.addEventListener("click", () => {
    const t = $(`${b.dataset.toggleTable}-table`);
    t.hidden = !t.hidden;
    b.textContent = t.hidden ? "Als Tabelle" : "Tabelle ausblenden";
  }));
  $("kanton").addEventListener("change", (e) => { state.kanton = e.target.value; loadMarkt(); });
  $("insurer").addEventListener("change", (e) => selectInsurer(+e.target.value));
  $("pp").addEventListener("input", onSlider);
  $("pp-reset").addEventListener("click", () => selectInsurer(state.insurer));
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { renderMarkt(); renderScenario(state.lastScenario || state.actual); renderQuality(); }, 120);
  });
  const links = [...document.querySelectorAll(".sections a[href^='#']")];
  const obs = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === `#${en.target.id}`));
  }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((sec) => obs.observe(sec));
}

async function init() {
  wire();
  const [meta, versicherer, kantone, vergleich] = await Promise.all(
    [api("/modell"), api("/versicherer"), api("/kantone"), api("/modellvergleich")]);
  Object.assign(state, { meta, versicherer, vergleich });
  state.horizon = state.qualityHorizon = Math.max(...meta.horizonte_verfuegbar);
  const latest = yearOf(state.horizon);
  $("headline").textContent = `Wer im Grundversicherungsmarkt ${latest} Versicherte gewinnt, und wer verliert.`;
  $("dek").textContent =
    `Prognose auf Basis der vom BAG genehmigten Prämien ${latest} und der Versichertenbestände bis ${meta.basisjahr}, ` +
    `für ${versicherer.length} Krankenversicherer in ${kantone.length} Kantonen. Modellversion ${meta.version}.`;
  const years = meta.horizonte_verfuegbar.map((h) => [h, `Prognose ${yearOf(h)}`]);
  segmented($("horizon-switch"), years, state.horizon, async (h) => {
    state.horizon = h; await loadMarkt(); await selectInsurer(state.insurer);
  });
  segmented($("quality-horizon"), meta.horizonte_verfuegbar.map((h) => [h, h === 1 ? "1 Jahr im Voraus" : `${h} Jahre im Voraus`]),
    state.qualityHorizon, (h) => { state.qualityHorizon = h; renderQuality(); });
  $("kanton").insertAdjacentHTML("beforeend", kantone.map((k) => `<option value="${k.kanton}">Kanton ${k.kanton}</option>`).join(""));
  $("insurer").innerHTML = versicherer.map((v) => `<option value="${v.bag_nr}">${v.name}</option>`).join("");
  renderQuality();
  await loadMarkt();
  await selectInsurer(versicherer[0].bag_nr);
}

init().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<p class="empty" style="margin-top:32px">Daten konnten nicht geladen werden (${err.message}). Läuft <code>marktradar train</code>?</p>`);
});
