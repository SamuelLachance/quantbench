/**
 * QuantBench — Worker de valorisation OTC À LA DEMANDE (Cloudflare).
 *
 * Port JS compact mais fidèle de la valorisation routée Damodaran (Python :
 * quantbench/valuation/{dcf,route,build_universal}.py + data/fmp.py). La clé FMP
 * est un SECRET du Worker (jamais exposée au client).
 *
 * GET https://<worker>/?ticker=NSRGY  ->  JSON profil de la fiche titre.
 *
 * Déploiement : voir worker/README.md (wrangler deploy + wrangler secret put FMP_API_KEY).
 */

const FMP = "https://financialmodelingprep.com/stable";
const ERP = 0.045;          // prime de risque actions
const RF = 0.042;           // taux sans risque (10Y ~ courant ; le build pré-construit utilise FRED live)

// ---------- utilitaires ----------
const num = (x) => {
  const v = typeof x === "string" ? parseFloat(x) : x;
  return Number.isFinite(v) ? v : null;
};
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const safeDiv = (a, b) => (a == null || !b ? null : a / b);
const B = 1e9;

async function fmpGet(path, key) {
  const r = await fetch(`${FMP}/${path}${path.includes("?") ? "&" : "?"}apikey=${key}`,
    { cf: { cacheTtl: 3600, cacheEverything: true } });
  if (!r.ok) throw new Error(`FMP ${r.status}`);
  return r.json();
}

// Actions privilégiées / bonds : mêmes règles que fmp._is_preferred (Python)
const PREF_NAME = /\bPFD\b|\bPREF(ERRED)?\s+(STOCK|SHARES|SHS|SEC|SERIES|EQUITY)|PERPETUAL\s+(RED\s+)?PREF|CUM\s+PERP|\bRATE\s+RESET\b|\bRST\s+PFD\b|\b(SUB(ORDINATED)?|JR|JUNIOR|SR|SENIOR)\s+(NOTES?|DEBENT)|\bDEBENTURES?\b|\bTRUST\s+PREF|\d+(\.\d+)?\s*%\s/i;
const PREF_TICKER = /-P[A-Z]{1,2}(\.(TO|V))?$/;
const isPreferred = (sym, name) => PREF_TICKER.test(sym) || (name && PREF_NAME.test(name));

const saneBeta = (b) => {
  b = num(b);
  return (b != null && b >= 0.1 && b <= 3.5) ? b : 1.1;
};

// ---------- trajectoires DCF ----------
function convergePath(start, end, n, convStart) {
  const hold = Math.min(Math.max(convStart - 1, 0), n);
  const path = new Array(n);
  for (let i = 0; i < hold; i++) path[i] = start;
  const ramp = n - hold;
  if (ramp === 1) path[hold] = end;
  else if (ramp > 1)
    for (let i = 0; i < ramp; i++) path[hold + i] = start + (end - start) * (i / (ramp - 1));
  return path;
}

function estimateGrowth(revenues) {
  const revs = revenues.filter((r) => r > 0);
  const n = revs.length;
  if (n < 2) return 0.05;
  const gLast = revs[n - 1] / revs[n - 2] - 1;
  const k3 = Math.min(3, n - 1);
  const gC3 = (revs[n - 1] / revs[n - 1 - k3]) ** (1 / k3) - 1;
  const gCf = (revs[n - 1] / revs[0]) ** (1 / (n - 1)) - 1;
  return clamp(0.5 * gLast + 0.3 * gC3 + 0.2 * gCf, -0.05, 0.45);
}

// DcfInputs -> résultat (port de dcf.value_dcf, mode réinvestissement ROIC)
function valueDcf(x) {
  const n = x.len1 + x.len2 + x.len3;
  const term = x.g3_end;
  const g = [
    ...convergePath(x.g1_begin, x.g1_end, x.len1, 1),
    ...convergePath(x.g2_begin, x.g2_end, x.len2, 1),
    ...convergePath(x.g3_begin, x.g3_end, x.len3, 1),
  ];
  const revenues = [], margins = convergePath(x.cur_margin, x.term_margin, n, x.margin_conv),
    taxes = convergePath(0.21, 0.25, n, 5),
    roicPath = convergePath(x.cur_roic, x.term_roic, n, 5);
  let rev = x.revenue_base;
  for (let i = 0; i < n; i++) { rev *= 1 + g[i]; revenues.push(rev); }
  // WACC : beta levé -> coût des FP ; WACC pondéré par la structure du capital
  const total = x.equity_value + x.debt_value;
  const wE = total === 0 ? 1 : x.equity_value / total, wD = total === 0 ? 0 : x.debt_value / total;
  const de = x.equity_value === 0 ? 0 : x.debt_value / x.equity_value;
  const betas = convergePath(x.unlev * (1 + 0.75 * de), clamp(x.unlev, 0.8, 1.2) * (1 + 0.75 * de), n, 5);
  const wacc = [], coe = [];
  for (let i = 0; i < n; i++) {
    const ce = RF + betas[i] * ERP;
    coe.push(ce);
    wacc.push(wE * ce + wD * (RF + 0.012) * (1 - 0.25));
  }
  // Flux : EBIT après impôt, réinvestissement lié au ROIC, FCFF
  const ebitAt = [], reinv = [], fcff = [];
  const curEbi = x.revenue_base * x.cur_margin * (1 - 0.21);
  for (let i = 0; i < n; i++) {
    const ebit = revenues[i] * margins[i];
    const eat = ebit * (1 - taxes[i]);
    ebitAt.push(eat);
    const prev = i === 0 ? curEbi : ebitAt[i - 1];
    const gEbi = prev > 0 ? eat / prev - 1 : g[i];
    let rr = roicPath[i] > 0 ? gEbi / roicPath[i] : 0;
    rr = clamp(rr, 0, 0.95);
    reinv.push(eat * rr);
    fcff.push(eat - eat * rr);
  }
  const disc = [];
  let acc = 1;
  for (let i = 0; i < n; i++) { acc *= 1 + wacc[i]; disc.push(acc); }
  let pvSum = 0;
  for (let i = 0; i < n; i++) pvSum += fcff[i] / disc[i];
  // valeur terminale
  const waccT = wacc[n - 1];
  if (waccT <= term) return null;                 // valeur terminale non définie
  const roicT = Math.min(x.term_roic, waccT + 0.02);
  const rrT = (term <= 0 || roicT === 0) ? 0 : term / roicT;
  const revT = revenues[n - 1] * (1 + term);
  const fcffT = revT * x.term_margin * (1 - 0.25) * (1 - rrT);
  const tv = fcffT / (waccT - term);
  const pvTv = tv / disc[n - 1];
  const valueOperating = pvSum + pvTv;
  const firm = valueOperating + x.cash;
  const equity = firm - x.debt_value;
  return { equity, firm, valueOperating, revenues, g, margins, ebitAt, reinv, fcff, wacc, coe, disc };
}

function buildInputs(fund, marginOverride, len) {
  const rev = fund.revenue;
  if (!rev || rev <= 0) return null;
  const revHist = (fund.revenue_history || []).filter((x) => x);
  const gStart = revHist.length >= 2 ? estimateGrowth(revHist) : 0.08;
  let m = marginOverride != null ? marginOverride
    : (fund.operating_margin != null ? fund.operating_margin : (safeDiv(fund.ebit, rev) ?? 0.10));
  m = clamp(m, -0.20, 0.75);
  const debt = fund.total_debt || 0, cash = fund.cash || 0, eqBook = fund.book_equity || 0;
  const mcap = fund.market_cap || rev;
  const invested = Math.max(eqBook + debt - cash, 0.05 * rev);
  const nopat = m * rev * 0.75;
  const curRoic = clamp(safeDiv(nopat, invested) ?? 0.12, 0.02, 0.60);
  const levBeta = fund.beta || 1.1;
  const de = safeDiv(debt, mcap) ?? 0;
  const unlev = de >= 0 ? levBeta / (1 + 0.75 * de) : levBeta;
  const costEq = RF + levBeta * ERP;
  const termRoic = clamp(costEq + 0.02, 0.07, Math.max(curRoic, 0.08));
  const term = Math.min(RF, 0.028);
  const L = len || [3, 4, 3];
  return {
    revenue_base: rev, g1_begin: gStart, g1_end: 0.8 * gStart + 0.2 * term,
    g2_begin: 0.8 * gStart + 0.2 * term, g2_end: 0.45 * gStart + 0.55 * term,
    g3_begin: 0.45 * gStart + 0.55 * term, g3_end: term,
    len1: L[0], len2: L[1], len3: L[2],
    cur_margin: m, term_margin: m, margin_conv: 3,
    cur_roic: curRoic, term_roic: termRoic,
    unlev, equity_value: mcap, debt_value: debt, cash,
    meta: { gStart, m, levBeta },
  };
}

// ---------- routage (port de route.py) ----------
function classify(fund) {
  const sec = (fund.sector || "").toLowerCase();
  const rev = fund.revenue, ebit = fund.ebit, ni = fund.net_income;
  if (rev == null || rev <= 0) return "actif_net";
  if (sec.includes("financial")) return "financiere";
  if (sec.includes("energy") || sec.includes("materials")) return "cyclique";
  if ((ebit != null && ebit < 0) || (ni != null && ni < 0)) return "jeune/deficitaire";
  return "standard";
}

function dcfEquity(fund, marginOverride) {
  const x = buildInputs(fund, marginOverride, [3, 4, 3]);
  if (!x) return null;
  const r = valueDcf(x);
  return r ? r.equity : null;
}

function valueRouted(fund, F) {
  const cat = classify(fund);
  let eq = null, method = "", conf = "moyenne";
  try {
    if (cat === "actif_net") {
      const be = fund.book_equity, cash = fund.cash || 0, debt = fund.total_debt || 0;
      const nav = (be != null && be > 0) ? be : cash - debt;
      if (nav > 0) { eq = nav; method = "Valeur d'actif net (pré-revenu / holding)"; conf = "faible"; }
    } else if (cat === "financiere") {
      const be = fund.book_equity, roe = fund.roe;
      if (be && be > 0 && roe != null) {
        let ke = RF + (fund.beta || 1.1) * ERP; const gg = Math.min(RF, 0.03); ke = Math.max(ke, gg + 0.01);
        const mult = clamp((roe - ke) / (ke - gg), -0.6, 4.0);
        eq = Math.max(be * (1 + mult), 0.2 * be);
        method = "Excess-return (capitaux propres — Damodaran financières)";
      }
    } else if (cat === "cyclique") {
      let nm = null;
      if (F && F.ebit && F.revenue) {
        const ms = []; for (let i = 0; i < F.ebit.length; i++)
          if (F.ebit[i] != null && F.revenue[i]) ms.push(F.ebit[i] / F.revenue[i]);
        if (ms.length) nm = ms.reduce((a, b) => a + b, 0) / ms.length;
      }
      eq = dcfEquity(fund, nm); method = "DCF sur bénéfices normalisés (cyclique)";
    } else if (cat === "jeune/deficitaire") {
      const om = fund.operating_margin;
      const target = (om != null && om > 0.05) ? om : 0.12;
      let base = dcfEquity(fund, target);
      if (base != null) {
        const ni = fund.net_income || 0, cash = fund.cash || 0;
        const burn = ni < 0 ? -ni : 0;
        const surv = burn > 0 ? clamp(0.3 + 0.15 * (cash / burn), 0.3, 0.9) : 0.85;
        const liq = 0.5 * (fund.book_equity || 0);
        eq = Math.max(base, 0) * surv + liq * (1 - surv);
      }
      method = "DCF top-down sur revenus (jeune) × survie"; conf = "faible";
    } else {
      eq = dcfEquity(fund, null); method = "DCF FCFF (standard)";
    }
  } catch (e) { eq = null; }
  // repli en cascade
  if (eq == null && fund.revenue > 0) { eq = dcfEquity(fund, null); method = "DCF FCFF (repli)"; }
  if (eq == null) {
    const be = fund.book_equity;
    if (be && be > 0) { eq = be; method = "Valeur comptable (repli)"; conf = "très faible"; }
    else return { ok: false, category: cat };
  }
  eq = Math.max(eq, 0);                          // responsabilité limitée
  const mcap = fund.market_cap, shares = fund.shares;
  return {
    ok: true, category: cat, method, confidence: conf,
    equity_value: Math.round(eq * 100) / 100,
    value_per_share: shares ? Math.round(eq * 1e9 / shares * 100) / 100 : null,
    price: fund.price, market_cap: mcap,
    upside: (mcap && mcap > 0) ? Math.round((eq / mcap - 1) * 10000) / 10000 : null,
  };
}

// projection 20 ans (pour le tableau + CSV) — mode ROIC, len 5/10/5
function projection(fund, marginOverride) {
  const x = buildInputs(fund, marginOverride, [5, 10, 5]);
  if (!x) return null;
  const r = valueDcf(x);
  if (!r) return null;
  const out = [];
  for (let i = 0; i < r.revenues.length; i++) {
    const ebit = r.revenues[i] * r.margins[i];
    out.push({
      year: i + 1, revenue: +r.revenues[i].toFixed(2),
      revenue_growth_pct: +(r.g[i] * 100).toFixed(2),
      operating_margin_pct: +(r.margins[i] * 100).toFixed(2),
      ebit: +ebit.toFixed(2), ebit_after_tax: +r.ebitAt[i].toFixed(2),
      reinvestment: +r.reinv[i].toFixed(2), fcff: +r.fcff[i].toFixed(2),
      wacc_pct: +(r.wacc[i] * 100).toFixed(2),
      cost_of_equity_pct: +(r.coe[i] * 100).toFixed(2),
      discount_factor: +r.disc[i].toFixed(4),
      pv_fcff: +(r.fcff[i] / r.disc[i]).toFixed(2),
    });
  }
  return out;
}

// ---------- fondamentaux depuis FMP ----------
function fundamentals(inc, bal, profile, quote) {
  if (!inc.length || !bal.length) return null;
  const i0 = inc[0], b0 = bal[0];
  const rev = num(i0.revenue), ebit = num(i0.operatingIncome), ni = num(i0.netIncome);
  let cash = num(b0.cashAndCashEquivalents), debt = num(b0.totalDebt);
  const ta = num(b0.totalAssets);
  if (ta && ta > 0) {                            // garde-fou données corrompues
    if (cash != null) cash = Math.min(cash, ta);
    if (debt != null) debt = Math.min(Math.max(debt, 0), ta);
  }
  const eq = num(b0.totalStockholdersEquity);
  const shares = num(i0.weightedAverageShsOutDil);
  const revHist = inc.map((r) => num(r.revenue)).filter((v) => v != null).reverse().map((v) => v / B);
  const p = profile || {}, q = quote || {};
  return {
    ticker: (p.symbol || "").toUpperCase(), name: p.companyName || p.symbol,
    sector: p.sector, industry: p.industry, summary: p.description,
    price: num(p.price) || num(q.price),
    market_cap: (num(p.marketCap) || num(q.marketCap) || 0) / B || null,
    shares, beta: saneBeta(p.beta),
    revenue: rev != null ? rev / B : null,
    revenue_history: revHist,
    ebit: ebit != null ? ebit / B : null, net_income: ni != null ? ni / B : null,
    total_debt: debt != null ? debt / B : null, cash: cash != null ? cash / B : null,
    book_equity: eq != null ? eq / B : null,
    operating_margin: (ebit && rev) ? ebit / rev : null,
    roe: (ni && eq) ? ni / eq : null,
  };
}

function financials(inc, bal) {
  const years = inc.map((r) => num(r.fiscalYear) || +String(r.date).slice(0, 4));
  const col = (arr, f, d = 1) => arr.map((r) => { const v = num(r[f]); return v != null ? v / (d * 1) : null; });
  return {
    years,
    revenue: inc.map((r) => num(r.revenue)),
    ebit: inc.map((r) => num(r.operatingIncome)),
    net_income: inc.map((r) => num(r.netIncome)),
  };
}

function statementsCard(inc, bal) {
  const g = (v) => v == null ? null : Math.round(v / B * 100) / 100;
  return {
    years: inc.map((r) => num(r.fiscalYear) || +String(r.date).slice(0, 4)),
    revenue: inc.map((r) => g(num(r.revenue))),
    ebit: inc.map((r) => g(num(r.operatingIncome))),
    net_income: inc.map((r) => g(num(r.netIncome))),
    total_assets: bal.map((r) => g(num(r.totalAssets))),
    equity: bal.map((r) => g(num(r.totalStockholdersEquity))),
    total_debt: bal.map((r) => g(num(r.totalDebt))),
  };
}

function resultsSummary(inc) {
  if (!inc.length) return null;
  const r0 = num(inc[0].revenue), r1 = inc[1] ? num(inc[1].revenue) : null, n0 = num(inc[0].netIncome);
  return {
    fiscal_year: num(inc[0].fiscalYear) || +String(inc[0].date).slice(0, 4),
    revenue: r0 ? +(r0 / B).toFixed(1) : null,
    rev_growth: (r0 && r1) ? +(r0 / r1 - 1).toFixed(4) : null,
    net_income: n0 ? +(n0 / B).toFixed(1) : null,
    net_margin: (n0 && r0) ? +(n0 / r0).toFixed(4) : null,
  };
}

// ---------- handler ----------
export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Cache-Control": "public, max-age=3600",
      "Content-Type": "application/json; charset=utf-8",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    const url = new URL(request.url);
    const ticker = (url.searchParams.get("ticker") || "").toUpperCase().trim();
    if (!ticker || !/^[A-Z0-9.\-]{1,12}$/.test(ticker))
      return new Response(JSON.stringify({ ok: false, error: "ticker invalide" }), { status: 400, headers: cors });
    const key = env.FMP_API_KEY;
    if (!key) return new Response(JSON.stringify({ ok: false, error: "clé FMP absente" }), { status: 500, headers: cors });

    try {
      const [inc, bal, profileArr, quoteArr, news] = await Promise.all([
        fmpGet(`income-statement?symbol=${ticker}&period=annual&limit=6`, key),
        fmpGet(`balance-sheet-statement?symbol=${ticker}&period=annual&limit=6`, key),
        fmpGet(`profile?symbol=${ticker}`, key),
        fmpGet(`quote?symbol=${ticker}`, key),
        fmpGet(`news/stock?symbols=${ticker}&limit=8`, key).catch(() => []),
      ]);
      const profile = Array.isArray(profileArr) ? profileArr[0] : profileArr;
      const quote = Array.isArray(quoteArr) ? quoteArr[0] : quoteArr;
      if (!inc.length || !bal.length || !profile)
        return json({ ok: false, ticker, error: "données financières indisponibles pour ce titre" }, cors);

      const fund = fundamentals(inc, bal, profile, quote);
      if (!fund || !fund.price)
        return json({ ok: false, ticker, error: "prix ou fondamentaux indisponibles" }, cors);
      // gate d'intégrité (comme build_one)
      if (!fund.shares || fund.shares < 100000 || !fund.market_cap || fund.market_cap < 0.002)
        return json({ ok: false, ticker, name: fund.name, error: "données non fiables (actions/cap implausibles)" }, cors);

      const F = financials(inc, bal);
      const val = valueRouted(fund, F);
      if (!val.ok) return json({ ok: false, ticker, name: fund.name, error: "valorisation impossible" }, cors);
      if (val.upside != null && val.upside > 5.0)
        return json({ ok: false, ticker, name: fund.name, error: "upside démesuré (donnée non fiable)" }, cors);

      // projection avec la marge de la catégorie
      let projMargin = null;
      if (val.category === "jeune/deficitaire") { const om = fund.operating_margin; projMargin = (om != null && om > 0.05) ? om : 0.12; }
      else if (val.category === "cyclique" && F.ebit && F.revenue) {
        const ms = []; for (let i = 0; i < F.ebit.length; i++) if (F.ebit[i] != null && F.revenue[i]) ms.push(F.ebit[i] / F.revenue[i]);
        if (ms.length) projMargin = ms.reduce((a, b) => a + b, 0) / ms.length;
      }
      const proj = (val.category === "actif_net" || val.category === "financiere") ? null : projection(fund, projMargin);

      const out = {
        ok: true, ticker, name: fund.name, sector: fund.sector, industry: fund.industry,
        summary: fund.summary, exchange: "OTC", on_demand: true,
        valuation: val,
        fundamentals: {
          price: fund.price, market_cap: fund.market_cap, shares: fund.shares, beta: fund.beta,
          revenue: fund.revenue, ebit: fund.ebit, net_income: fund.net_income,
          total_debt: fund.total_debt, cash: fund.cash, book_equity: fund.book_equity,
          operating_margin: fund.operating_margin, roe: fund.roe,
        },
        statements: statementsCard(inc, bal), results_summary: resultsSummary(inc),
        projection: proj, montecarlo: null, forensics: null, shortterm: null,
        news: (Array.isArray(news) ? news : []).slice(0, 8).map((a) => ({
          title: a.title, url: a.url, publisher: a.publisher || a.site, publishedDate: a.publishedDate || a.date,
        })),
        documents: [], report_url: null, ars_pdf_url: null, filing_url: null, pdf_url: null,
      };
      return json(out, cors);
    } catch (e) {
      return json({ ok: false, ticker, error: "erreur de calcul : " + (e.message || e) }, cors, 500);
    }
  },
};

function json(obj, cors, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: cors });
}
