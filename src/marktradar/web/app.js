"use strict";
// Marktradar – Oberfläche. Bewusst ohne Framework und ohne CDN: läuft offline,
// im Firmennetz und bleibt in einer Datei nachvollziehbar.

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

const state = { markt: [], versicherer: [], modell: null, filter: "gross", curve: [], insurer: null };

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

// ---------- Tooltip ----------
const tip = document.getElementById("tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - r.width - 8);
  const top = y + r.height + 20 > window.innerHeight ? y - r.height - 12 : y + 14;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---------- 01 Markt ----------
function marktRows() {
  return state.filter === "gross" ? state.markt.filter((r) => r.bestand_basis >= 100000) : state.markt;
}

function renderMarkt() {
  const rows = marktRows();
  const host = document.getElementById("markt-chart");
  host.innerHTML = "";
  const lo = Math.min(-1, ...rows.map((r) => r.anteilswachstum_untergrenze_pct));
  const hi = Math.max(1, ...rows.map((r) => r.anteilswachstum_obergrenze_pct));
  const plotW = host.getBoundingClientRect().width - (window.innerWidth <= 600 ? 176 : 302);
  const ticks = niceTicks(lo, hi, plotW < 260 ? 3 : 6);
  const domain = [Math.min(lo, ticks[0]), Math.max(hi, ticks[ticks.length - 1])];

  const axis = el("div", { class: "dp-axis" }, host);
  el("span", { text: "Versicherer" }, axis);
  const axisCell = el("div", {}, axis);
  el("span", { class: "dp-val", text: "Prognose" }, axis);

  const rowEls = rows.map((r) => {
    const row = el("div", { class: "dp-row" }, host);
    const name = el("button", { class: "dp-name", type: "button", text: r.name, title: r.name }, row);
    name.addEventListener("click", () => selectInsurer(r.bag_nr, true));
    const cell = el("div", {}, row);
    el("span", { class: "dp-val", text: `${fmtSigned(r.anteilswachstum_pct)} %` }, row);
    row.addEventListener("mousemove", (e) => showTip(
      `<b>${r.name}</b><br>Marktanteil ${nf(2).format(r.marktanteil_basis_pct)} % (${r.basisjahr})<br>` +
      `Prognose ${r.prognosejahr}: <b>${fmtSigned(r.anteilswachstum_pct)} %</b><br>` +
      `<span class="muted">80 %: ${fmtSigned(r.anteilswachstum_untergrenze_pct)} bis ${fmtSigned(r.anteilswachstum_obergrenze_pct)} %</span><br>` +
      `Bestand ${fmtInt(r.bestand_basis)} → ${fmtInt(r.bestand_prognose)}`, e.clientX, e.clientY));
    row.addEventListener("mouseleave", hideTip);
    return { r, cell };
  });

  const width = axisCell.getBoundingClientRect().width || 400;
  const pad = 8;
  const x = (v) => pad + ((v - domain[0]) / (domain[1] - domain[0])) * (width - 2 * pad);

  const axSvg = el("svg:svg", { width, height: 26, class: "axis" }, axisCell);
  for (const t of ticks) {
    el("svg:text", { x: x(t), y: 17, "text-anchor": "middle", text: t === 0 ? "0" : `${fmtSigned(t, 0)} %` }, axSvg);
  }

  for (const { r, cell } of rowEls) {
    const s = el("svg:svg", { width, height: 26 }, cell);
    for (const t of ticks) el("svg:line", { x1: x(t), x2: x(t), y1: 0, y2: 26, class: t === 0 ? "zeroline" : "gridline" }, s);
    el("svg:line", {
      x1: x(r.anteilswachstum_untergrenze_pct), x2: x(r.anteilswachstum_obergrenze_pct), y1: 13, y2: 13,
      stroke: css("--range"), "stroke-width": 2, "stroke-linecap": "round",
    }, s);
    el("svg:circle", {
      cx: x(r.anteilswachstum_pct), cy: 13, r: 5,
      fill: r.anteilswachstum_pct >= 0 ? css("--gain") : css("--loss"),
      stroke: css("--paper"), "stroke-width": 2,
    }, s);
  }

  document.getElementById("markt-table").innerHTML = table(
    ["Versicherer", "Marktanteil %", "Prognose %", "Untergrenze %", "Obergrenze %", "Bestand", "Bestand Prognose"],
    rows.map((r) => [r.name, nf(2).format(r.marktanteil_basis_pct), fmtSigned(r.anteilswachstum_pct),
      fmtSigned(r.anteilswachstum_untergrenze_pct), fmtSigned(r.anteilswachstum_obergrenze_pct),
      fmtInt(r.bestand_basis), fmtInt(r.bestand_prognose)]));
}

function table(head, rows) {
  const th = head.map((h, i) => `<th class="${i ? "n" : ""}">${h}</th>`).join("");
  const tb = rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "n" : ""}">${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`;
}

// ---------- 02 Szenario ----------
async function selectInsurer(bagNr, scroll = false) {
  state.insurer = bagNr;
  document.getElementById("insurer").value = String(bagNr);
  if (scroll) document.getElementById("szenario").scrollIntoView();
  state.curve = await api("/elastizitaet", { bag_nr: bagNr, von_pp: -5, bis_pp: 5, schritt_pp: 0.5 });
  renderScenario();
}

function currentPoint() {
  const pp = +document.getElementById("pp").value;
  return state.curve.find((p) => Math.abs(p.praemienanpassung_rel_pp - pp) < 1e-6) || state.curve[10];
}

function renderScenario() {
  const p = currentPoint();
  if (!p) return;
  document.getElementById("pp-out").textContent = fmtSigned(p.praemienanpassung_rel_pp, 1);
  const delta = p.bestand_prognose - p.bestand_basis;
  document.getElementById("figures").innerHTML = [
    figure(`Marktanteil ${p.basisjahr}`, `${nf(2).format(p.marktanteil_basis_pct)} %`, `${fmtInt(p.bestand_basis)} Versicherte`),
    figure(`Veränderung Marktanteil ${p.prognosejahr}`, `${fmtSigned(p.anteilswachstum_pct)} %`,
      `80 %-Intervall ${fmtSigned(p.anteilswachstum_untergrenze_pct)} bis ${fmtSigned(p.anteilswachstum_obergrenze_pct)} %`),
    figure(`Bestand ${p.prognosejahr}`, fmtInt(p.bestand_prognose), `${fmtSigned(delta, 0)} Personen, bei konstanter Marktgrösse`),
  ].join("");
  renderCurve(p);
}

function figure(label, value, sub) {
  return `<div class="figure"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function renderCurve(selected) {
  const host = document.getElementById("curve");
  host.innerHTML = "";
  const W = host.getBoundingClientRect().width || 600;
  const H = 300, m = { t: 12, r: 16, b: 34, l: 56 };
  const pts = state.curve.map((p) => {
    const base = p.bestand_basis;
    const toPersons = (pct) => base * pct / 100;
    return { pp: p.praemienanpassung_rel_pp, y: p.bestand_prognose - p.bestand_basis,
      lo: toPersons(p.anteilswachstum_untergrenze_pct), hi: toPersons(p.anteilswachstum_obergrenze_pct), p };
  });
  const yMin = Math.min(0, ...pts.map((d) => d.lo)), yMax = Math.max(0, ...pts.map((d) => d.hi));
  const yt = niceTicks(yMin, yMax, 5);
  const y0 = Math.min(yMin, yt[0]), y1 = Math.max(yMax, yt[yt.length - 1]);
  const x = (v) => m.l + ((v + 5) / 10) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const s = el("svg:svg", { width: W, height: H, role: "img",
    "aria-label": "Veränderung des Bestands nach Prämienanpassung" }, host);
  const ax = el("svg:g", { class: "axis" }, s);
  for (const t of yt) {
    el("svg:line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? "zeroline" : "gridline" }, ax);
    el("svg:text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: fmtCompact(t) }, ax);
  }
  for (let t = -5; t <= 5; t += 1) {
    el("svg:text", { x: x(t), y: H - 12, "text-anchor": "middle", text: t === 0 ? "0" : fmtSigned(t, 0) }, ax);
  }
  el("svg:text", { x: W - m.r, y: H, "text-anchor": "end", text: "%-Punkte vs. Markt" }, ax);

  const area = pts.map((d) => `${x(d.pp)},${y(d.hi)}`).concat(pts.slice().reverse().map((d) => `${x(d.pp)},${y(d.lo)}`));
  el("svg:polygon", { points: area.join(" "), fill: css("--band") }, s);
  el("svg:polyline", { points: pts.map((d) => `${x(d.pp)},${y(d.y)}`).join(" "), fill: "none",
    stroke: css("--series-1"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);

  const sel = pts.find((d) => d.p === selected);
  if (sel) {
    el("svg:line", { x1: x(sel.pp), x2: x(sel.pp), y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1 }, s);
    el("svg:circle", { cx: x(sel.pp), cy: y(sel.y), r: 6, fill: css("--series-1"), stroke: css("--paper"), "stroke-width": 2 }, s);
    const right = x(sel.pp) > W * 0.7;
    // Label auf die Seite, auf der die Kurve nicht verläuft (fällt sie, darüber rechts; sonst darunter)
    const below = sel.y < 0;
    el("svg:text", { x: x(sel.pp) + (right ? -12 : 12), y: y(sel.y) + (below ? 20 : -12), "text-anchor": right ? "end" : "start",
      class: "direct-label", text: `${fmtSigned(sel.y, 0)} Personen` }, s);
  }

  const cross = el("svg:line", { y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, s);
  const hit = el("svg:rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, s);
  hit.addEventListener("mousemove", (e) => {
    const bx = s.getBoundingClientRect().left;
    const pp = ((e.clientX - bx - m.l) / (W - m.l - m.r)) * 10 - 5;
    const d = pts.reduce((a, b) => (Math.abs(b.pp - pp) < Math.abs(a.pp - pp) ? b : a));
    cross.setAttribute("x1", x(d.pp)); cross.setAttribute("x2", x(d.pp)); cross.setAttribute("visibility", "visible");
    showTip(`<b>${fmtSigned(d.pp, 1)} %-Punkte</b> vs. Markt<br>Marktanteil ${fmtSigned(d.p.anteilswachstum_pct)} %<br>` +
      `Bestand ${fmtSigned(d.y, 0)} Personen<br><span class="muted">80 %: ${fmtCompact(d.lo)} bis ${fmtCompact(d.hi)}</span>`,
      e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("click", (e) => {
    const bx = s.getBoundingClientRect().left;
    const pp = Math.round((((e.clientX - bx - m.l) / (W - m.l - m.r)) * 10 - 5) * 2) / 2;
    document.getElementById("pp").value = String(Math.max(-5, Math.min(5, pp)));
    renderScenario();
  });
}

// ---------- 03 Modellgüte ----------
function renderQuality() {
  const meta = state.modell;
  const b = meta.backtest;
  document.getElementById("quality-figures").innerHTML = [
    figure("Genauer als die einfache Annahme", `${nf(0).format(b.verbesserung_vs_konstant * 100)} %`,
      `kleinerer Fehler als „Marktanteil bleibt gleich“, über ${b.testjahre} Testjahre`),
    figure("Gewinner und Verlierer", nf(2).format(b.rangkorrelation),
      "Rangkorrelation: 1 = Reihenfolge perfekt, 0 = Zufall"),
    figure("Unsicherheit ehrlich angegeben", `${nf(0).format(b.intervall_abdeckung * 100)} %`,
      `der echten Werte im 80 %-Intervall (vor Kalibrierung ${nf(0).format(b.intervall_abdeckung_roh * 100)} %)`),
  ].join("");
  renderErrorChart(meta.backtest_pro_jahr);
  document.getElementById("guete-table").innerHTML = table(
    ["Prognosejahr", "Fehler Modell %", "Fehler „bleibt gleich“ %", "Fehler „wie Vorjahr“ %", "Rangkorrelation", "Intervall-Treffer %"],
    meta.backtest_pro_jahr.map((r) => [r.basisjahr + 1, nf(1).format(r.mae_modell * 100), nf(1).format(r.mae_konstant * 100),
      nf(1).format(r.mae_vorjahr * 100), nf(2).format(r.rangkorrelation), nf(0).format(r.intervall_abdeckung * 100)]));
  renderLive(meta.live_monitoring || []);
  const facts = [
    ["Modellversion", meta.version],
    ["Trainiert auf", `Basisjahre ${meta.trainiert_auf_basisjahre[0]}–${meta.trainiert_auf_basisjahre[1]} (${fmtInt(meta.n_trainingszeilen)} Versicherer-Jahre)`],
    ["Datenquelle", `<a href="${meta.daten_quelle}">BAG Aufsichtsdaten OKP</a>`],
    ["Daten-Prüfsumme", `<code>${(meta.daten_sha256 || "–").slice(0, 12)}</code>`],
    ["Code-Stand", meta.git_commit ? `<code>${meta.git_commit}</code>` : "–"],
    ["Modell", "Gradient Boosting, monotone Prämien-Constraints, Conformal-Intervalle"],
  ];
  document.getElementById("facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderErrorChart(rows) {
  const host = document.getElementById("error-chart");
  host.innerHTML = "";
  const W = host.getBoundingClientRect().width || 600;
  const H = 260, m = { t: 14, r: 120, b: 30, l: 40 };
  const series = [
    { key: "mae_modell", name: "Modell", color: css("--series-1") },
    { key: "mae_konstant", name: "bleibt gleich", color: css("--series-2") },
  ];
  const years = rows.map((r) => r.basisjahr + 1);
  const vmax = Math.max(...rows.flatMap((r) => series.map((s) => r[s.key] * 100)));
  const yt = niceTicks(0, vmax, 4);
  const top = Math.max(vmax, yt[yt.length - 1]);
  const x = (yr) => m.l + ((yr - years[0]) / (years[years.length - 1] - years[0])) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - v / top) * (H - m.t - m.b);

  const s = el("svg:svg", { width: W, height: H, role: "img", "aria-label": "Prognosefehler je Jahr" }, host);
  const ax = el("svg:g", { class: "axis" }, s);
  for (const t of yt) {
    el("svg:line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? "zeroline" : "gridline" }, ax);
    el("svg:text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: `${nf(0).format(t)} %` }, ax);
  }
  const step = W < 520 ? 3 : 1;
  years.forEach((yr, i) => {
    if (i % step === 0 || i === years.length - 1) el("svg:text", { x: x(yr), y: H - 8, "text-anchor": "middle", text: yr }, ax);
  });
  for (const se of series) {
    el("svg:polyline", { points: rows.map((r) => `${x(r.basisjahr + 1)},${y(r[se.key] * 100)}`).join(" "),
      fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);
    for (const r of rows) {
      el("svg:circle", { cx: x(r.basisjahr + 1), cy: y(r[se.key] * 100), r: 4, fill: se.color, stroke: css("--paper"), "stroke-width": 2 }, s);
    }
    const last = rows[rows.length - 1];
    el("svg:text", { x: W - m.r + 10, y: y(last[se.key] * 100) + 4, class: "direct-label",
      text: `${se.name} ${nf(1).format(last[se.key] * 100)} %` }, s);
  }
  const cross = el("svg:line", { y1: m.t, y2: H - m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, s);
  const hit = el("svg:rect", { x: m.l - 10, y: m.t, width: W - m.l - m.r + 20, height: H - m.t - m.b, fill: "transparent" }, s);
  hit.addEventListener("mousemove", (e) => {
    const bx = s.getBoundingClientRect().left;
    const r = rows.reduce((a, b) => (Math.abs(x(b.basisjahr + 1) - (e.clientX - bx)) < Math.abs(x(a.basisjahr + 1) - (e.clientX - bx)) ? b : a));
    cross.setAttribute("x1", x(r.basisjahr + 1)); cross.setAttribute("x2", x(r.basisjahr + 1)); cross.setAttribute("visibility", "visible");
    showTip(`<b>Prognose für ${r.basisjahr + 1}</b><br>Modell: ${nf(1).format(r.mae_modell * 100)} %<br>` +
      `„Bleibt gleich“: ${nf(1).format(r.mae_konstant * 100)} %<br><span class="muted">Rangkorrelation ${nf(2).format(r.rangkorrelation)} · ${r.n} Versicherer</span>`,
      e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

function renderLive(records) {
  const host = document.getElementById("live");
  if (!records.length) {
    host.innerHTML = `<p class="empty">Noch keine Live-Prüfung. Sobald das BAG ein neues Datenjahr publiziert, ` +
      `misst <code>marktradar update</code> das bisherige Modell an Jahren, die es nie gesehen hat, und trainiert danach neu. ` +
      `Die Resultate erscheinen hier.</p>`;
    return;
  }
  host.innerHTML = table(["Prognosejahr", "Modellversion", "Fehler Modell %", "Fehler „bleibt gleich“ %", "Rangkorr."],
    records.map((r) => [r.prognosejahr, `<code>${r.modell_version}</code>`, nf(1).format(r.mae_modell * 100),
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
    const t = document.getElementById(`${b.dataset.toggleTable}-table`);
    t.hidden = !t.hidden;
    b.textContent = t.hidden ? "Als Tabelle" : "Tabelle ausblenden";
  }));
  document.getElementById("insurer").addEventListener("change", (e) => selectInsurer(+e.target.value));
  document.getElementById("pp").addEventListener("input", renderScenario);
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { renderMarkt(); renderCurve(currentPoint()); renderErrorChart(state.modell.backtest_pro_jahr); }, 120);
  });
  const links = [...document.querySelectorAll(".sections a[href^='#']")];
  const obs = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === `#${en.target.id}`));
  }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((sec) => obs.observe(sec));
}

async function init() {
  wire();
  const [markt, versicherer, modell] = await Promise.all([api("/markt"), api("/versicherer"), api("/modell")]);
  Object.assign(state, { markt, versicherer, modell });
  const first = markt[0];
  document.getElementById("headline").textContent =
    `Wer im Grundversicherungsmarkt ${first.prognosejahr} Versicherte gewinnt, und wer verliert.`;
  document.getElementById("dek").textContent =
    `Prognose für ${first.prognosejahr} auf Basis der öffentlichen Aufsichtsdaten des BAG bis ${first.basisjahr}, ` +
    `für ${versicherer.length} Krankenversicherer. Modellversion ${modell.version}.`;
  const sel = document.getElementById("insurer");
  sel.innerHTML = versicherer.map((v) => `<option value="${v.bag_nr}">${v.name}</option>`).join("");
  renderMarkt();
  renderQuality();
  await selectInsurer(versicherer[0].bag_nr);
}

init().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<p class="empty" style="margin-top:32px">Daten konnten nicht geladen werden (${err.message}). Läuft <code>marktradar train</code>?</p>`);
});
