/* QuantBench — helpers partagés (pro) */
const QB = {
  nf0: new Intl.NumberFormat('fr-FR', {maximumFractionDigits: 0}),
  nf1: new Intl.NumberFormat('fr-FR', {minimumFractionDigits: 1, maximumFractionDigits: 1}),
  nf2: new Intl.NumberFormat('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
  pct(v, d = 1) {
    if (v == null || isNaN(v)) return '—';
    return (v >= 0 ? '+' : '−') + (d === 0 ? this.nf0 : this.nf1).format(Math.abs(v * 100)) + ' %';
  },
  usd(v, d = 2) { return (v == null || isNaN(v)) ? '—' : (d === 0 ? this.nf0 : this.nf2).format(v) + ' $'; },
  bn(v) { return (v == null || isNaN(v)) ? '—' : this.nf1.format(v) + ' Md$'; },
  qs(n) { return new URLSearchParams(location.search).get(n); },

  initTheme() {
    try { const t = localStorage.getItem('qb-theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) {}
    const b = document.getElementById('themeBtn');
    if (b) b.addEventListener('click', () => {
      const r = document.documentElement;
      const dark = getComputedStyle(r).getPropertyValue('--bg').trim().startsWith('#0');
      r.setAttribute('data-theme', dark ? 'light' : 'dark');
      try { localStorage.setItem('qb-theme', dark ? 'light' : 'dark'); } catch (e) {}
    });
  },

  async universe() {
    if (this._uni) return this._uni;
    try {
      const d = await (await fetch('us/_screener.json', {cache: 'no-store'})).json();
      this._uni = (d.rows || []).concat(d.suspects || []);
    } catch (e) { this._uni = []; }
    return this._uni;
  },

  async mountSearch() {
    const inp = document.getElementById('tickerSearch');
    if (!inp) return;
    const uni = await this.universe();
    const dl = document.getElementById('tickerList');
    if (dl) dl.innerHTML = uni.map(r => `<option value="${r.ticker}">${r.name || ''}</option>`).join('');
    const go = () => {
      const t = (inp.value || '').trim().toUpperCase().split(/\s|—/)[0];
      if (t) location.href = 'stock.html?t=' + encodeURIComponent(t);
    };
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
    inp.addEventListener('change', go);
    const btn = document.getElementById('searchGo');
    if (btn) btn.addEventListener('click', go);
  },

  downloadCSV(rows, filename) {
    if (!rows || !rows.length) return;
    const cols = Object.keys(rows[0]);
    const esc = v => {
      if (v == null) return '';
      const s = String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const csv = [cols.join(',')].concat(
      rows.map(r => cols.map(c => esc(r[c])).join(','))).join('\n');
    const blob = new Blob(['﻿' + csv], {type: 'text/csv;charset=utf-8'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  },

  /* ---- graphiques SVG (sans dépendance) ---- */
  _scale(vals, lo, hi) { const mn = Math.min(...vals), mx = Math.max(...vals); const r = (mx - mn) || 1;
    return v => hi - (v - mn) / r * (hi - lo); },

  cone(el, st) {                              // cône court terme p10/p90 + p50
    if (!el || !st || !st.points) { if (el) el.innerHTML = '<div class="muted">—</div>'; return; }
    const P = st.points, W = 640, H = 200, pad = 30;
    const xs = P.map(p => p.d), all = P.flatMap(p => [p.p10, p.p90]);
    const x = this._scale(xs, pad, W - 8); const xf = v => pad + (v - xs[0]) / ((xs[xs.length - 1] - xs[0]) || 1) * (W - pad - 8);
    const y = this._scale(all, 12, H - 22);
    const up = P.map(p => `${xf(p.d).toFixed(1)},${y(p.p90).toFixed(1)}`).join(' ');
    const dn = P.slice().reverse().map(p => `${xf(p.d).toFixed(1)},${y(p.p10).toFixed(1)}`).join(' ');
    const mid = P.map(p => `${xf(p.d).toFixed(1)},${y(p.p50).toFixed(1)}`).join(' ');
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">
      <text x="${(W/2).toFixed(0)}" y="11" text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="700" font-family="var(--mono)">80 % de probabilité</text>
      <polygon points="${up} ${dn}" fill="var(--accent-soft)"/>
      <polyline points="${mid}" fill="none" stroke="var(--accent)" stroke-width="2"/>
      <text x="${pad}" y="${H - 6}" class="ax">aujourd'hui</text>
      <text x="${W - 8}" y="${H - 6}" text-anchor="end" class="ax">+${st.days} j</text>
      <text x="${xf(xs[xs.length-1]).toFixed(0)}" y="${y(P[P.length-1].p90).toFixed(0)}" text-anchor="end" class="ax">${QB.usd(P[P.length-1].p90,0)}</text>
      <text x="${xf(xs[xs.length-1]).toFixed(0)}" y="${(y(P[P.length-1].p10)+10).toFixed(0)}" text-anchor="end" class="ax">${QB.usd(P[P.length-1].p10,0)}</text>
    </svg>`;
  },

  histogram(el, bins, refx, medx) {           // histogramme Monte Carlo
    if (!el || !bins || !bins.length) { if (el) el.innerHTML = ''; return; }
    const W = 640, H = 190, pad = 8, mB = 16;
    const xs = bins.map(b => b.x), ys = bins.map(b => b.y);
    const xmin = xs[0], xmax = xs[xs.length - 1], ymax = Math.max(...ys) || 1;
    const X = v => pad + (v - xmin) / ((xmax - xmin) || 1) * (W - 2 * pad);
    const bw = (W - 2 * pad) / bins.length * 0.92;
    let s = bins.map(b => { const h = b.y / ymax * (H - mB - 6), x = X(b.x) - bw / 2, y = H - mB - h;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="var(--accent)" opacity="0.45"/>`; }).join('');
    const line = (v, c, lbl, anc) => (v == null || v < xmin || v > xmax) ? '' :
      `<line x1="${X(v).toFixed(1)}" y1="0" x2="${X(v).toFixed(1)}" y2="${H - mB}" stroke="${c}" stroke-width="2" stroke-dasharray="4 3"/><text x="${(X(v) + (anc === 'end' ? -4 : 4)).toFixed(1)}" y="11" text-anchor="${anc || 'start'}" class="ax" fill="${c}">${lbl}</text>`;
    s += line(medx, 'var(--accent)', 'médiane') + line(refx, 'var(--ink)', 'cours', 'end');
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">${s}</svg>`;
  },

  bars(el, proj, key, color) {                // barres de projection (ex. revenue)
    if (!el || !proj || !proj.length) { if (el) el.innerHTML = ''; return; }
    const W = 640, H = 180, pad = 22, n = proj.length, bw = (W - pad - 6) / n * 0.72;
    const vals = proj.map(p => p[key]); const mx = Math.max(...vals) || 1;
    const bars = proj.map((p, i) => {
      const h = Math.max(1, p[key] / mx * (H - pad - 14));
      const x = pad + i * (W - pad - 6) / n, yv = H - pad - h;
      return `<rect x="${x.toFixed(1)}" y="${yv.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5" fill="${color || 'var(--accent)'}" opacity="${0.45 + 0.55 * i / n}"/>`;
    }).join('');
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart">${bars}
      <text x="${pad}" y="${H - 6}" class="ax">an 1</text>
      <text x="${W - 6}" y="${H - 6}" text-anchor="end" class="ax">an ${n}</text></svg>`;
  },
};
