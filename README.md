# QuantBench — valorisation intrinsèque, honnêtement

[![CI](https://github.com/SamuelLachance/quantbench/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelLachance/quantbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Valorisation intrinsèque de **~16 800 sociétés** — NASDAQ, NYSE, TSX, TSXV et gré à gré —
selon la méthode d'Aswath Damodaran, reconstruite chaque jour. Chaque titre reçoit une
**valeur intrinsèque** et une **note de risque de A+ à F**.

⚠️ Outil de recherche **éducatif** — ceci n'est **pas** un conseil d'investissement.

## Le principe

Deux questions distinctes, deux réponses distinctes.

**Combien vaut cette société ?** Pas de DCF universel : le flux de trésorerie disponible
n'a aucun sens pour une banque, dont la dette est la matière première, ni pour une
foncière, dont les amortissements immobiliers écrasent le résultat sans faire sortir un
euro. Chaque titre est donc **routé** vers la méthode que Damodaran prescrit pour son cas.

| route | méthode | pourquoi |
|---|---|---|
| financière | rendement excédentaire, côté équité | la valeur d'entreprise n'a pas de sens quand la dette est un intrant |
| foncière | FFO capitalisé | l'amortissement immobilier est comptable, pas économique |
| service public régulé | bénéfices capitalisés côté équité | rentabilité fixée par le régulateur |
| cyclique | marge moyenne **du cycle** | un exercice isolé ne dit rien d'une mine |
| mature en perte | bénéfices normalisés × probabilité de redressement | une perte passagère n'est pas une faillite |
| jeune | descendant sur le chiffre d'affaires × probabilité de survie | pas d'historique à normaliser |
| détresse | pondérée par la probabilité de défaut + liquidation | |
| pré-revenu, holding | actif net **réalisable** | aucun flux à actualiser |
| standard | DCF FCFF | |

**Quelle est la probabilité de perdre durablement sa mise ?** C'est une autre question, et
un titre peut être très décoté *parce qu'il est* très dangereux. La note de risque la
traite séparément, sur huit dimensions — solvabilité, autonomie de trésorerie, rentabilité
de cycle, volatilité, dilution, liquidité, **confiance dans la donnée**, mur de
refinancement.

## Trois règles qui structurent tout le dépôt

**Un contrôle rejette une donnée fausse, jamais un résultat qui déplaît.** Quatre seuils
de validation confrontaient la capitalisation — grandeur d'actionnaire — au chiffre
d'affaires et au résultat opérationnel — grandeurs d'entreprise — sans jamais ajouter la
dette : ils mesuraient donc le **levier**. Charter Communications, 19,6 Md$ de
capitalisation pour 12,7 Md$ de résultat opérationnel, était écartée au seul motif qu'elle
porte 95,8 Md$ de dette. Supprimés.

**Aucun seuil d'opinion dans le code.** Les repères sont **mesurés** sur l'univers réel :
bêta désendetté, marge, ventes sur capital, ROIC et taux de récupération par industrie
(`scripts/build_industry_stats.py`), quantiles de risque par dimension
(`scripts/build_risk_stats.py`). Un test refuse l'introduction d'une constante qui
dirait ce qu'est une *bonne* couverture d'intérêts.

**Toute correction se termine par un test.** La suite encode **287 invariants** —
361 cas de test une fois les paramétrages développés — chacun correspondant à un
défaut réellement observé en production : `test_invariants.py` pour le moteur,
`test_donnees_universal.py` pour la couche de données, `test_dcf.py` et
`test_eval.py` pour les calculs.
C'est le test, et non le correctif, qui rend la chose permanente.

## Ce que la discipline a trouvé

Quelques défauts corrigés, tous au niveau de la **méthode** et jamais du titre :

- **La décote de liquidation portait sur l'équité au lieu de l'actif.** Elle revenait à
  supposer que les dettes subissent la même décote au bénéfice de l'actionnaire. L'erreur
  vaut (1 − taux) × passif : nulle sans dette — d'où son invisibilité — et maximale
  précisément là où la liquidation se pose.
- **La détresse se mesurait en résultat comptable.** Branicks publie −288,7 M€ d'EBIT pour
  +54,8 M€ de flux d'exploitation, la perte étant une réévaluation IFRS de son parc. Nous
  envoyions cette foncière à la liquidation.
- **Un bénéfice jamais encaissé finançait une croissance gratuite.** FDCTech affiche sur
  dix ans 3 M$ de résultat cumulé pour 31 M$ de trésorerie **consommée**.
- **Une marge est un rapport de sommes, pas la moyenne de rapports.** Pop Culture a triplé
  son chiffre d'affaires en perdant de l'argent : ses marges anciennes, gagnées sur un
  dixième du volume, dominaient la moyenne.
- **Un historique en pesos traité comme des dollars.** La valeur d'Edenor était
  surestimée d'exactement le taux de change.
- **279 Md$ de foncières absentes de l'univers** parce que le fournisseur marque `isFund`
  toute société dont la raison sociale contient « Trust ».

## Structure

```
quantbench/
  valuation/  route.py        routage sectoriel + les neuf méthodes
              build_universal.py  moteur DCF, bêta ascendant, primes pays et taille
              industry_stats.json repères MESURÉS par industrie et secteur
  risk/       dimensions.py   les huit dimensions de risque
              score.py        agrégation, plafonds, treize grades
              risk_calibration.json  quantiles gelés, datés, versionnés
  data/       fmp.py          univers, fondamentaux, change
              validate.py     contrôles d'entrée — identités comptables uniquement
              repair.py       réparation avant renoncement
  forensics/  Altman, Beneish, Piotroski — signaux statistiques neutres
scripts/      build_site_fmp.py     build quotidien, 5 shards parallèles
              build_industry_stats.py · build_risk_stats.py   calibrages
              check_build.py        garde-fou qui BLOQUE un déploiement défectueux
              mesurer_les_notes.py  juge la note de risque contre les faits
app/          index.html · stock.html · screener.html · shortterm.html
tests/        287 invariants hors ligne, 382 cas (invariants, données, DCF, éval.)
```

## Lancement

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Construire le site (clé FMP requise) :

```bash
FMP_API_KEY=... python scripts/build_site_fmp.py --limit 200
```

Recalibrer les repères sectoriels et les quantiles de risque :

```bash
FMP_API_KEY=... python scripts/build_industry_stats.py
FMP_API_KEY=... python scripts/build_risk_stats.py --limite 3000
```

Servir le site :

```bash
python -m http.server 8765 --directory app
```

## Ce que le modèle ne sait pas, et le dit

- **Les poids de la note de risque sont uniformes**, et ce n'est pas un oubli. Nos
  fondamentaux ne sont pas historisés : corréler les notes d'aujourd'hui aux cours passés
  mesurerait un biais, pas un pouvoir prédictif. Le build **archive les notes chaque
  mois** ; `scripts/mesurer_les_notes.py` les confrontera aux faits dans douze mois et
  refuse explicitement de conclure si l'intervalle de confiance contient 0,50.
- **Le signal court terme ne publie aucune probabilité** tant qu'une calibration hors
  échantillon n'a pas démontré de pouvoir prédictif. Elle n'en a pas démontré.
- **Certaines coquilles de gré à gré n'ont pas de fait daté de radiation** — ni SEC, ni
  liste du fournisseur. Elles sont publiées avec la date de leurs derniers comptes plutôt
  que cachées, et l'érosion temporelle les ramène à leur valeur réelle.
- **Une note excellente n'empêche aucune perte.** Elle mesure la fragilité *observable
  dans les comptes* ; ce qu'ils ne contiennent pas, elle ne le voit pas.

## Licence

[MIT](LICENSE) © 2026 Samuel Lachance. Outil de recherche éducatif, sans conseil
d'investissement.
