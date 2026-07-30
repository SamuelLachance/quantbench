"""Reparation automatique des donnees d'entree.

Detecter une anomalie sans tenter de la corriger ne fait que reduire la couverture.
Chaque anomalie identifiee par `validate.valider` recoit ici une strategie de
REPARATION fondee sur une source alternative ou une identite comptable. On ne
renonce a un titre qu'apres l'echec de la reparation, et le motif reste explicite.

    donnees brutes -> valider -> reparer -> revalider -> valoriser OU rejeter

Reparations implementees, chacune repondant a une anomalie CONSTATEE :

  capitalisation d'une ligne ADR   -> capitalisation de la cotation D'ORIGINE
                                      (Kalbe Farma : 8,7 M$ -> 1,75 Md$)
  identite du bilan violee         -> fonds propres RECALCULES = actif - passif
  comptes perimes                  -> reconstitution des douze derniers mois a
                                      partir des trimestres
  taux de change introuvable       -> passage par une devise pivot (EUR)

Chaque reparation appliquee est TRACEE : la fiche du titre indique ce qui a ete
corrige et par quel moyen, pour que rien ne soit fait dans le dos du lecteur.
"""

from __future__ import annotations

from . import fmp


def _champ_imputable(fund, F):
    """Quand actif != passif + fonds propres, UN des trois nombres est faux — mais
    l'identite seule ne dit pas lequel. On le determine par l'EXERCICE PRECEDENT,
    source independante et deja chargee : le champ corrompu est celui dont la valeur
    RECALCULEE par l'identite colle a l'annee passee alors que la valeur PUBLIEE s'en
    ecarte. On ne repare que si un seul champ repond a ce critere.

    Reparer globalement en posant fonds propres = actif - passif serait pire que le
    mal : l'identite boucle alors PAR CONSTRUCTION et le controle s'eteint pour
    toujours. Mesure sur l'univers, cette voie attribuait 258,93 Md$ de fonds propres
    a une societe de 11,6 M$ de capitalisation, dont l'ecart de bilan atteint 1 210 %
    et n'est imputable a aucun champ : sa valorisation ressortait a +16 598 %, et seul
    le maintien du motif l'empechait d'etre publiee.

    Retourne (nom_du_champ, valeur_corrigee) ou None."""
    ta, tl = fund.get("total_assets"), fund.get("total_liab")
    fp = fund.get("total_equity")
    if not ta or ta <= 0 or tl is None or fp is None:
        return None
    if abs(ta - (tl + fp)) / ta <= 0.05:
        return None
    precedent = _exercice_precedent(F, fund)
    if not precedent:
        return None
    candidats = []
    for cle, publie, recalcule in (("total_assets", ta, tl + fp),
                                   ("total_liab", tl, ta - fp),
                                   ("total_equity", fp, ta - tl)):
        ref = precedent.get(cle)
        if ref is None or abs(ref) < 1e-12:
            continue
        ecart_publie = abs(publie - ref) / abs(ref)
        ecart_recalcule = abs(recalcule - ref) / abs(ref)
        # Le champ est impute quand la valeur recalculee colle a l'an passe (moins de
        # 25 % d'ecart) alors que la publiee s'en eloigne d'au moins le double.
        if ecart_recalcule < 0.25 and ecart_publie > 2 * max(ecart_recalcule, 0.05):
            candidats.append((cle, recalcule))
    return candidats[0] if len(candidats) == 1 else None


def _exercice_precedent(F, fund):
    """Valeurs de l'exercice N-1, ramenees dans l'UNITE DE `fund` (dollars, milliards).

    `F` est publie en devise locale et en unites, `fund` en dollars et en milliards :
    les comparer directement reviendrait a confronter des roupies a des dollars. Le
    facteur se DEDUIT de l'exercice courant, qui figure dans les deux et provient de
    la meme ligne source — il vaut donc exactement le change, que la valeur soit
    corrompue ou non."""
    if not F or not fund:
        return None
    courant, serie = fund.get("total_assets"), F.get("total_assets") or []
    if not courant or not serie or not serie[0]:
        return None
    k = courant / serie[0]
    out = {}
    for cle, nom in (("total_assets", "total_assets"), ("total_liab", "total_liab"),
                     ("total_equity", "equity")):
        v = F.get(nom) or []
        if len(v) >= 2 and v[1] is not None:
            out[cle] = v[1] * k
    return out or None


def _ttm_depuis_trimestres(symbol):
    """Douze derniers mois reconstitues a partir des comptes trimestriels, quand
    le dernier exercice annuel est perime."""
    try:
        inc = fmp._json(f"income-statement?symbol={symbol}&period=quarter&limit=4")
        bal = fmp._json(f"balance-sheet-statement?symbol={symbol}&period=quarter&limit=1")
    except Exception:
        return None
    if not inc or len(inc) < 4 or not bal:
        return None
    # Le trimestriel doit etre RECENT. Sans ce controle, quatre trimestres clos en
    # 2017 produisaient de paisibles "douze mois glissants" en 2026 : la reparation
    # rajeunissait des comptes de neuf ans au lieu de les signaler.
    from datetime import datetime, timezone
    dernier = str(inc[0].get("date") or "")[:10]
    try:
        clos = datetime.strptime(dernier, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if (datetime.now(timezone.utc) - clos).days > 400:
        return None
    som = lambda k: sum((fmp._num(q.get(k)) or 0.0) for q in inc[:4])
    return {"revenue": som("revenue"), "operatingIncome": som("operatingIncome"),
            "netIncome": som("netIncome"),
            "shares": fmp._num(inc[0].get("weightedAverageShsOutDil")),
            # La DEVISE de publication des trimestres : les montants reconstitues
            # doivent etre convertis comme le reste, faute de quoi les comptes d'une
            # societe non americaine entrent dans le modele libelles en dollars.
            "devise": inc[0].get("reportedCurrency"),
            "bilan": bal[0], "date": inc[0].get("date")}


def _taux_via_pivot(devise):
    """Taux introuvable en direct : on passe par l'euro, tres largement cote."""
    if not devise or devise.upper() == "USD":
        return None
    c = devise.upper()
    try:
        j = fmp._json(f"quote?symbol={c}EUR")
        p = fmp._num(j[0].get("price")) if j else None
        eur = fmp.fx_to_usd("EUR")
        if p and p > 0 and eur:
            return p * eur
    except Exception:
        pass
    return None


def reparer(symbol, fund, F, entry, motifs):
    """Tente de corriger les anomalies detectees. Retourne la liste des
    reparations appliquees (le dictionnaire `fund` est modifie sur place)."""
    faites = []
    if not fund or not motifs:
        return faites
    texte = " | ".join(motifs)

    # 1. Bilan qui ne boucle pas -> imputation du champ fautif
    #    Cette reparation ecrivait `book_equity` alors que le controle lit
    #    `total_equity` : elle n'effacait donc jamais son propre motif. Cent
    #    soixante-quinze titres le portaient, un seul le franchissait.
    if "identite du bilan" in texte:
        impute = _champ_imputable(fund, F)
        if impute:
            cle, valeur = impute
            if cle == "total_equity" and fund.get("book_equity") is not None:
                # `book_equity` est la part ATTRIBUABLE : on lui reporte la correction
                # en preservant l'ecart des minoritaires et des privilegiees.
                minoritaires = (fund.get("total_equity") or 0.0) - fund["book_equity"]
                fund["book_equity"] = valeur - minoritaires
            fund[cle] = valeur
            faites.append(f"{cle} impute par l'identite du bilan "
                          f"(seul champ contredisant l'exercice precedent)")

    # 2. Taux de change introuvable -> devise pivot
    if "taux de change indisponible" in texte:
        t = _taux_via_pivot(fund.get("financial_currency"))
        if t:
            for k in ("revenue", "ebit", "net_income", "total_debt", "cash",
                      "book_equity", "total_assets", "dep_amort", "cfo"):
                if fund.get(k) is not None:
                    fund[k] = fund[k] * t
            fund["revenue_history"] = [v * t for v in (fund.get("revenue_history") or [])]
            fund["fx_indisponible"] = False
            faites.append(f"taux {fund.get('financial_currency')} obtenu via l'euro")

    # 3. Poste de flux hors d'echelle -> on ECARTE le champ plutot que la societe
    #    (une donnee non fiable vaut mieux absente : la methode s'en passe).
    for cle, libelle in (("dep_amort", "amortissements"), ("cfo", "flux d'exploitation")):
        if libelle in texte and fund.get(cle) is not None:
            fund[cle] = None
            faites.append(f"{libelle} ecartes (valeur hors d'echelle)")

    # 4. Comptes perimes -> douze derniers mois trimestriels
    if "comptes perimes" in texte:
        ttm = _ttm_depuis_trimestres(symbol)
        if ttm and ttm.get("revenue"):
            # Le taux de change etait FIGE A 1. Les comptes trimestriels sont publies
            # dans la devise de la societe : reconstituer douze mois glissants sans
            # les convertir revenait a faire entrer des roupies ou des wons dans le
            # modele comme s'il s'agissait de dollars — l'erreur valant exactement le
            # taux de change. C'est le meme defaut que celui qui portait la valeur de
            # l'argentine Edenor a +6 864 %, ici loge dans la REPARATION elle-meme,
            # donc invisible : il ne frappait que des titres deja signales.
            devise = ttm.get("devise") or fund.get("financial_currency")
            fx = fmp.fx_to_usd(devise) if devise else 1.0
            if not fx or fx <= 0:
                return faites                    # sans taux fiable, on ne repare pas
            B = 1e9
            fund["revenue"] = ttm["revenue"] * fx / B
            if ttm.get("operatingIncome") is not None:
                fund["ebit"] = ttm["operatingIncome"] * fx / B
                if fund["revenue"]:
                    fund["operating_margin"] = fund["ebit"] / fund["revenue"]
            if ttm.get("netIncome") is not None:
                fund["net_income"] = ttm["netIncome"] * fx / B
            fund["date_ttm"] = ttm.get("date")
            faites.append(f"comptes reconstitues sur douze mois glissants "
                          f"(dernier trimestre {ttm.get('date')})")

    return faites


__all__ = ["reparer"]
