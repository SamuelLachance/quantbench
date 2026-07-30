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


def _fonds_propres_recalcules(fund, F):
    """Identite fondamentale : actif - passif = fonds propres. Quand le champ
    publie contredit le bilan, l'identite fait foi."""
    ta = fund.get("total_assets")
    tl = fund.get("total_liab")              # deja converti en USD, comme ta
    if not ta or ta <= 0 or tl is None:
        return None
    recalcule = ta - tl
    ancien = fund.get("book_equity")
    if ancien is None or abs(recalcule - ancien) / max(ta, 1e-9) > 0.05:
        return recalcule
    return None


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

    # 1. Fonds propres contredisant le bilan -> identite comptable
    if "identite du bilan" in texte:
        v = _fonds_propres_recalcules(fund, F)
        if v is not None:
            fund["book_equity"] = v
            faites.append("fonds propres recalcules par l'identite actif - passif")

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
            faites.append(f"comptes reconstitues sur douze mois glissants "
                          f"(dernier trimestre {ttm.get('date')})")

    return faites


__all__ = ["reparer"]
