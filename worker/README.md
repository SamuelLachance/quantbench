# QuantBench — Worker de valorisation OTC (à la demande)

Le site est statique (GitHub Pages) et ne peut pas exécuter la valorisation Python.
Pour les titres **OTC** (marché de gré à gré, ~9 500 titres), la fiche est générée
**à la demande** par ce Cloudflare Worker quand l'utilisateur cherche le ticker.

- `quantbench-otc.js` — port JavaScript compact et fidèle de la valorisation routée
  Damodaran (mêmes catégories, garde-fous, plancher d'équité que le pipeline Python).
- La clé FMP est un **secret du Worker** (jamais exposée au client).
- Réponse : le même JSON qu'une fiche pré-construite → `stock.html` la rend telle quelle.

## Déploiement (une seule fois, ~5 min)

Prérequis : Node.js installé, et un compte Cloudflare (le même que pour le DNS de quantbench.ca).

```bash
npm install -g wrangler        # CLI Cloudflare
cd worker
wrangler login                 # ouvre le navigateur, connecte ton compte Cloudflare
wrangler secret put FMP_API_KEY   # colle ta clé FMP quand demandé (reste côté serveur)
wrangler deploy                # déploie + crée otc.quantbench.ca (DNS + cert auto)
```

Le Worker est ensuite live sur **https://otc.quantbench.ca** — l'URL déjà configurée
dans `app/assets/app.js` (`QB.OTC_API`). Aucune autre étape côté site.

## Test

```bash
curl "https://otc.quantbench.ca/?ticker=NSRGY"   # Nestlé (ADR OTC)
```

Réponse attendue : `{"ok":true,"ticker":"NSRGY","valuation":{...},...}`.
Un titre sans données ou aux fondamentaux corrompus renvoie `{"ok":false,"error":"…"}`,
que la fiche affiche proprement.

## Notes

- Le Worker met les réponses FMP en cache 1 h (`cf.cacheTtl`) → rapide et économe en quota.
- Le taux sans risque est fixé à 4,2 % (le build pré-construit utilise le taux FRED live) ;
  d'où un écart de quelques points de % avec les titres pré-construits — attendu.
- Pour changer l'URL (ex. sous-domaine workers.dev au lieu du domaine custom), édite
  `routes` dans `wrangler.toml` **et** `QB.OTC_API` dans `app/assets/app.js`.
