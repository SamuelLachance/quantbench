/* QuantBench — helpers partagés */
const QB = {
  nf0: new Intl.NumberFormat('fr-FR', {maximumFractionDigits: 0}),
  nf1: new Intl.NumberFormat('fr-FR', {minimumFractionDigits: 1, maximumFractionDigits: 1}),
  nf2: new Intl.NumberFormat('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
  pct(v, d = 1) {
    if (v == null || isNaN(v)) return '—';
    const n = (d === 0 ? this.nf0 : this.nf1).format(Math.abs(v * 100));
    return (v >= 0 ? '+' : '−') + n + ' %';
  },
  usd(v, d = 2) {
    if (v == null || isNaN(v)) return '—';
    return (d === 0 ? this.nf0 : this.nf2).format(v) + ' $';
  },
  bn(v) {                                     // milliards
    if (v == null || isNaN(v)) return '—';
    return this.nf1.format(v) + ' Md$';
  },
  initTheme() {
    const btn = document.getElementById('themeBtn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const root = document.documentElement;
      const dark = getComputedStyle(root).getPropertyValue('--bg').trim().startsWith('#0');
      root.setAttribute('data-theme', dark ? 'light' : 'dark');
      try { localStorage.setItem('qb-theme', dark ? 'light' : 'dark'); } catch (e) {}
    });
    try {
      const t = localStorage.getItem('qb-theme');
      if (t) document.documentElement.setAttribute('data-theme', t);
    } catch (e) {}
  },
  qs(name) {
    return new URLSearchParams(location.search).get(name);
  },
};
