# QuantBench — banc d'essai de valorisation honnête

Valorisation intrinsèque probabiliste (DCF FCFF multi-phase à la Damodaran + Monte Carlo),
sur **données réelles gratuites** (SEC EDGAR, FRED, marché). Inspiré de la discipline du fonds
Medallion : ne pas chercher une fausse précision, mais mesurer honnêtement l'incertitude.

## Structure

```
quantbench/
  valuation/   dcf.py (moteur FCFF, numpy pur) + montecarlo.py (copule gaussienne)
  data/        edgar.py (SEC XBRL) + market.py (prix/bêta/taux) + build.py (assemblage)
  service.py   payload complet d'un ticker (CLI + API)
  api/app.py   FastAPI : /api/value/{ticker}, /api/screener + pages statiques
app/           index.html (page ticker) + screener.html (NASDAQ) + data.json/screener.json
scripts/       value_ticker.py, batch_screener.py, demo_valuation.py
tests/         test_dcf.py (contrôle par forme fermée)
```

## Installation & lancement

```bash
pip install -r requirements.txt
```

Valoriser un titre (écrit `app/data.json`) :
```bash
python scripts/value_ticker.py MSFT
```

Screener d'un univers (écrit `app/screener.json`) :
```bash
python scripts/batch_screener.py            # NASDAQ par défaut
```

Site live (API + pages) :
```bash
uvicorn quantbench.api.app:app --reload --port 8000
# http://localhost:8000  (ticker)  ·  /screener.html  (NASDAQ)  ·  /api/value/NVDA
```

Volet court terme (banc d'essai honnête, anti-overfitting) :
```bash
python scripts/shortterm_demo.py BTC-USD
# grid-search mean-reversion -> Deflated Sharpe : signal réel ou fluke ?
```

Recherche de ticker en live : lancez le backend, ouvrez `http://localhost:8000`,
tapez un symbole (ex. `NVDA`) dans la barre — la valorisation est calculée à la volée.

Tests :
```bash
python -m pytest tests/ -q      # 16 tests (DCF + eval + backtest)
```

## Méthode & limites

- **Croissance** calibrée sur l'historique récent (mélange dernier YoY / CAGR 3 ans / CAGR complet, bornée).
- **Réinvestissement** lié au ROIC déclinant (mode Damodaran), pas seulement au ratio ventes/capital.
- **ERP** implicite fixe (~4,5 %), pas le rendement passé du marché (biais rétrospectif corrigé).
- Le cas de base est **conservateur** : sur ces hypothèses, la plupart des mégacaps sont richement valorisées.
- Univers limité aux **grandes capitalisations non-financières** (le DCF FCFF ne convient ni aux banques/assureurs
  ni aux sociétés déficitaires). Les splits d'actions peuvent fausser la capitalisation → résultats « suspects » isolés.

> ⚠️ Outil de recherche éducatif — **pas un conseil d'investissement**.
