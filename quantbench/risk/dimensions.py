"""Les huit dimensions de risque, mesurees sur les donnees deja chargees.

Chaque fonction retourne un SIGNAL BRUT oriente risque — plus il est grand, plus le
titre est risque — ou None quand la dimension n'est pas definissable. Aucune ne
retourne de note : la conversion en rang se fait dans `score.py`, contre une table de
quantiles gelee.

Aucun seuil de valeur n'apparait ici. Les seules constantes sont des minima
ARITHMETIQUES (il faut trois points pour un ecart interquartile) ou des bornes de
winsorisation dont l'effet sur le rang est nul par monotonie.
"""

from __future__ import annotations

import math
from ..bilan import est_une_activite_de_bilan
from ..series import paires_voisines, portee, renseigne

# Nombre minimal d'exercices pour un ecart interquartile : contrainte arithmetique,
# pas une opinion.
_MIN_EXERCICES_IQR = 3
# Fenetre de normalisation des benefices, en exercices. Damodaran prescrit une
# couverture d'interets normalisee pour toute activite volatile : un instantane fait
# perdre plusieurs crans a un cyclique en creux, puis les lui rend l'annee suivante.
_FENETRE_NORMALISATION = 3


def _mediane(valeurs):
    v = sorted(x for x in valeurs if x is not None and math.isfinite(x))
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def _quartiles(valeurs):
    v = sorted(x for x in valeurs if x is not None and math.isfinite(x))
    if len(v) < _MIN_EXERCICES_IQR:
        return None

    def q(p):
        pos = p * (len(v) - 1)
        bas = int(math.floor(pos))
        haut = min(bas + 1, len(v) - 1)
        return v[bas] + (v[haut] - v[bas]) * (pos - bas)

    return q(0.75) - q(0.25)


def _borne(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Regime : la nature du bilan decide de la regle appliquee
# --------------------------------------------------------------------------- #
def regime(fund, F=None, seuil_amortissement=None) -> str:
    """Nature economique du bilan. Determine par une MESURE STRUCTURELLE, jamais par
    un libelle sectoriel — Apollo est etiquetee "Asset Management" et detient un
    assureur, FDCTech est etiquetee "Financial Services" et vend des logiciels.

    Cinq regimes :
      financiere               le bilan EST l'outil de production ; les interets sont
                               une matiere premiere et le flux d'exploitation suit les
                               encours de credit, pas l'exploitation
      amortissement_lourd      l'amortissement domine le resultat : l'EBIT est un
                               artefact comptable (une fonciere est apres
                               amortissements immobiliers, elle encaisse pourtant ses
                               loyers)
      sans_dette_beneficiaire  aucune charge d'interets ET benefice : la solvabilite
                               ne se pose pas
      sans_dette_deficitaire   aucune charge d'interets ET perte : la solvabilite est
                               INDEFINIE, non pas excellente. C'est le piege central
                               de cette dimension — la population sans interets est
                               majoritairement composee de coquilles sans chiffre
                               d'affaires, a qui une regle naive decernerait la
                               meilleure note possible
      exploitante              le cas general
    """
    # Test STRUCTUREL, ecrit ici et non emprunte au routage : `_financiere_de_bilan`
    # y retourne True par DEFAUT quand le chiffre d'affaires est nul, ce qui est sans
    # consequence la-bas — elle n'y est appelee que sur le secteur financier — mais
    # ferait de toute societe pre-revenu une banque. Wesizwe Platinum, minier sans
    # production, en relevait. Le simple rapport actif/chiffre d'affaires ne suffit
    # pas davantage : il attrape les foncieres, dont le bilan est lourd sans etre
    # pour autant leur outil de credit — Prologis en relevait aussi.
    # Ce qui fait la financiere de bilan est la conjonction d'un LEVIER extreme et
    # d'un bilan lourd : les banques tiennent 8 a 10 % de fonds propres, les
    # assureurs 5 a 15 %, une fonciere 40 a 60 %.
    # Le ROUTAGE a deja tranche la nature de l'activite, et sa decision est testee :
    # une fonciere ou un service public regule releve de l'amortissement lourd, quelle
    # que soit la minceur de ses fonds propres. Simon Property ne tient que 9 % de
    # fonds propres — non parce qu'elle prete de l'argent, mais parce que ses centres
    # commerciaux sont amortis depuis des decennies — et le test de levier seul en
    # faisait une financiere de bilan.
    from ..valuation.route import classify
    try:
        if classify(fund, None, F) in ("fonciere", "reglementee"):
            return "amortissement_lourd"
    except Exception:                                  # noqa: BLE001
        pass
    ind = (fund.get("industry") or "").lower()
    ta, rev, be = fund.get("total_assets"), fund.get("revenue"), fund.get("book_equity")
    # SOURCE UNIQUE, partagee avec le routage de valorisation : une meme societe
    # ne peut pas etre une banque pour l'un et un industriel pour l'autre.
    if est_une_activite_de_bilan(ta, be, rev):
        return "financiere"
    # Mots-cles STRICTEMENT non ambigus. "Capital Markets" et "Credit Services"
    # designent aussi bien un preteur qu'un reseau de paiement ou un editeur de
    # logiciels : Visa et Mastercard portent le second, FDCTech le premier. Les
    # inclure classait Visa en financiere de bilan — l'erreur meme que le routage
    # s'echine a eviter, et qui la sortait a -73 %.
    if any(k in ind for k in ("bank", "insurance", "mortgage", "thrift")):
        return "financiere"
    interets = fund.get("interest_expense")
    ebit = fund.get("ebit")
    if not interets:
        return "sans_dette_beneficiaire" if (ebit or 0) > 0 else "sans_dette_deficitaire"
    da = fund.get("dep_amort")
    if da and ebit is not None:
        denom = ebit + da
        # SEUIL POSE, ET IL FAUT LE DIRE. Ce commentaire affirmait que le seuil
        # etait « MESURE sur l'univers (percentile ecrit dans le calibrage) ». Il
        # ne l'a jamais ete : la cle `seuil_amortissement` n'existe pas dans
        # `risk_calibration.json` et aucun script ne l'ecrit, si bien que
        # `(None or 0.40)` vaut 0,40 sur tout chemin d'execution et pour tout
        # titre. Le code documentait l'inverse de ce qu'il faisait, dans le module
        # meme qui fonde la distinction entre mesure et convention.
        # Ce qui EST justifie, c'est la forme : c'est l'intensite d'amortissement
        # qui declenche la bascule vers l'excedent avant amortissements, pas une
        # liste de secteurs. La valeur 0,40, elle, est une convention — au sens
        # ou l'est deja `seuil_base_actionnaire`, dont le calibrage explique que
        # la mesure a ete tentee, a echoue, et que la valeur est assumee.
        # Le calibrage reste prioritaire s'il vient un jour a porter la cle.
        if denom > 0 and (da / denom) > (seuil_amortissement or _SEUIL_AMORTISSEMENT_POSE):
            return "amortissement_lourd"
    return "exploitante"


# --------------------------------------------------------------------------- #
# D1 — Solvabilite : capacite a honorer les charges financieres
# --------------------------------------------------------------------------- #
def d1_solvabilite(fund, F, reg):
    """Signal = -couverture des interets. Le meilleur predicteur isole du defaut.

    La couverture se mesure sur la MEDIANE des exercices disponibles et non sur le
    dernier : un cyclique en creux perdrait plusieurs crans une annee pour les
    reprendre la suivante, ce qui ruinerait la stabilite meme de la note.

    Pas de rang sectoriel : le defaut est ABSOLU. Une biotech qui ne couvre pas ses
    interets n'est pas rassuree par le fait que ses consoeurs non plus."""
    if reg == "sans_dette_beneficiaire":
        # MODALITE, pas une mesure : -inf designe le meilleur cas possible de la
        # dimension. Retourner 0.0 la classait au-dessus de toute societe couvrant
        # ses interets — Apple, qui n'en paie aucun, ressortait au 63e centile de
        # RISQUE de solvabilite.
        return -math.inf, "aucune charge d'interets, exploitation beneficiaire"
    if reg == "sans_dette_deficitaire":
        return None, "solvabilite indefinie : aucune dette, mais aucun benefice"
    if reg == "financiere":
        # Une banque ne se juge pas a sa couverture d'interets — ils sont son cout
        # de production. Elle se juge a son COUSSIN DE FONDS PROPRES TANGIBLES : un
        # ecart d'acquisition n'absorbe aucune perte.
        ta = fund.get("total_assets")
        be = fund.get("book_equity")
        if not ta or ta <= 0 or be is None:
            return None, None
        tangible = be - (fund.get("goodwill_intangibles") or 0.0)
        return -(tangible / ta), "fonds propres tangibles rapportes a l'actif"

    ebits = list((F or {}).get("ebit") or [])
    interets = list((F or {}).get("interest_expense") or [])
    if reg == "amortissement_lourd":
        das = list((F or {}).get("dep_amort") or [])
        ebits = [(e + d) if (e is not None and d is not None) else e
                 for e, d in zip(ebits, das + [None] * len(ebits))]
    paires = [(e, i) for e, i in zip(ebits, interets)
              if e is not None and i is not None and i > 0]
    if len(paires) >= _MIN_EXERCICES_IQR:
        paires = paires[:_FENETRE_NORMALISATION]
        couv = _mediane([e / i for e, i in paires])
    else:
        e = fund.get("ebit")
        if reg == "amortissement_lourd" and fund.get("dep_amort") is not None:
            e = (e or 0.0) + fund["dep_amort"]
        i = fund.get("interest_expense")
        couv = (e / i) if (e is not None and i) else None
    if couv is None:
        return None, None
    libelle = ("couverture des interets par l'excedent avant amortissements"
               if reg == "amortissement_lourd" else "couverture des interets")
    return -_borne(couv, -50.0, 50.0), libelle


# --------------------------------------------------------------------------- #
# D2 — Autonomie de tresorerie
# --------------------------------------------------------------------------- #
def consommation_selon_le_regime(fund, reg):
    """Tresorerie consommee, corrigee du CAPEX DE CROISSANCE.

    Le flux libre brut est un mauvais juge pour une activite tres capitalistique : un
    service public regule finance ses investissements par la dette PAR CONSTRUCTION,
    avec un rendement garanti par le regulateur, et une fonciere en developpement fait
    de meme. Leur flux libre est negatif en permanence sans que rien ne menace.
    Duke Energy et Southern Company ressortaient plafonnees a D+ pour "moins d'un an
    d'autonomie", et E.ON pour "passif non couvert".

    On ne retient donc que le capex de MAINTENANCE, approxime par la dotation aux
    amortissements — la part de l'investissement qui remplace l'outil existant, par
    opposition a celle qui l'agrandit. L'approximation est declaree ; elle vaut
    uniquement pour les regimes ou l'amortissement domine le resultat."""
    from ..valuation.route import consommation_de_tresorerie
    if reg != "amortissement_lourd":
        return consommation_de_tresorerie(fund)
    capex, da = fund.get("capex"), fund.get("dep_amort")
    if capex is None or da is None:
        return consommation_de_tresorerie(fund)
    return consommation_de_tresorerie(dict(fund, capex=-min(abs(capex), abs(da))))


def d2_autonomie(fund, F, reg):
    """Signal = -(annees d'autonomie). Une annee de tresorerie vaut une annee
    partout : aucune normalisation sectorielle. Que la consommation soit la NORME en
    biotech est le resultat attendu, pas un biais a corriger — normaliser dans le
    secteur reviendrait a declarer saine une societe parce que ses voisines brulent
    aussi."""
    if reg == "financiere":
        # Le flux d'exploitation d'une banque suit ses encours de credit, pas son
        # exploitation : JPMorgan convertit 0,41 sans la moindre anomalie.
        return None, None
    # « AUCUN TABLEAU DE FLUX » N'EST PAS « L'EXPLOITATION S'AUTOFINANCE ».
    # `consommation_de_tresorerie` rend 0,0 dans les deux cas, et cette dimension
    # lisait ce zero comme la meilleure nouvelle possible : une coquille sans
    # comptes decrochait -inf, soit le meilleur rang de tout l'univers, devant
    # Apple. Une dimension non definissable doit dire qu'elle ne l'est pas — son
    # poids se redistribue alors sur les autres, ce que le module sait faire.
    from ..valuation.route import consommation_est_mesurable
    if not consommation_est_mesurable(fund):
        return None, None
    conso = consommation_selon_le_regime(fund, reg)
    if conso <= 0:
        return -math.inf, "l'exploitation finance ses investissements"
    cash = max(fund.get("cash") or 0.0, 0.0)
    return -_borne(cash / conso, 0.0, 50.0), "annees d'autonomie de tresorerie"


# --------------------------------------------------------------------------- #
# D3 — Rentabilite structurelle de cycle
# --------------------------------------------------------------------------- #
def d3_rentabilite(fund, F, reg):
    """Signal = -(marge de cycle). Rang SECTORIEL : c'est l'une des deux seules
    dimensions ou le relatif est legitime — 4 % de marge est excellent en
    distribution alimentaire, mediocre en logiciel. La marge n'est pas un verdict de
    survie mais un ecart a la norme du metier."""
    from ..valuation.route import _roe_normalise, marge_de_cycle
    if reg == "financiere":
        roe = _roe_normalise(fund, F)
        return (None, None) if roe is None else (-roe, "rentabilite des fonds propres")
    if reg == "amortissement_lourd":
        ni, da = fund.get("net_income"), fund.get("dep_amort")
        ta = fund.get("total_assets")
        if ni is None or da is None or not ta or ta <= 0:
            return None, None
        return -((ni + da) / ta), "flux recurrent rapporte a l'actif"
    revenus = [r for r in ((F or {}).get("revenue") or []) if r]
    if not revenus:
        # Aucun chiffre d'affaires : ce n'est pas une donnee manquante mais une
        # MODALITE, et elle designe le maximum de risque de la dimension.
        return math.inf, "aucun chiffre d'affaires"
    m = marge_de_cycle(F)
    if m is None:
        return None, None
    return -_borne(m, -2.0, 1.0), "marge d'exploitation moyenne du cycle"


# --------------------------------------------------------------------------- #
# D4 — Volatilite d'exploitation
# --------------------------------------------------------------------------- #
def d4_volatilite(fund, F, reg):
    """Signal = dispersion de l'exploitation, par ECART INTERQUARTILE.

    L'ecart-type est interdit ici : une marge se calcule sur un chiffre d'affaires
    qui peut tendre vers zero, sa queue n'est donc pas bornee et un seul exercice
    degenere fixerait la mesure. L'ecart interquartile est insensible aux extremes.

    Rang SECTORIEL : la cyclicite d'une miniere n'est pas un defaut de gestion."""
    from ..valuation.route import _hist_margins
    composantes = []
    marges = [_borne(m, -1.0, 1.0) for m in _hist_margins(F)]
    iqr_marge = _quartiles(marges)
    if iqr_marge is not None:
        composantes.append(iqr_marge)
    # LA SERIE NE SE PURGE PAS AVANT D'EN TIRER DES CROISSANCES ANNUELLES.
    # Elle etait refermee sur ses trous, puis lue par indices voisins : un exercice
    # manquant fabriquait une pseudo-croissance de DEUX ans, qui gonfle l'ecart
    # interquartile et donc la volatilite mesuree. La profondeur exigee se comptait
    # de surcroit sur la serie PURGEE, donc surestimee.
    revenus = list((F or {}).get("revenue") or [])
    positif = lambda r: renseigne(r) and r > 0
    if sum(1 for r in revenus if positif(r)) >= _MIN_EXERCICES_IQR + 1:
        croissances = [math.log(a / b) for a, b in paires_voisines(revenus, positif)]
        iqr_croissance = _quartiles(croissances)
        if iqr_croissance is not None:
            composantes.append(iqr_croissance)
    # LES DEUX COMPOSANTES, OU AUCUNE. La moyenne etait divisee par le NOMBRE de
    # composantes disponibles — un ou deux selon la profondeur des comptes — alors
    # que la table de centiles gelee est mesuree sur une population d'au moins six
    # exercices, donc presque entierement a DEUX composantes. Une societe aux
    # comptes courts etait ainsi classee contre une loi qui n'est pas la sienne, et
    # ressortait plus favorablement : rang median 0,419 contre 0,510, mesure sur
    # 281 societes.
    #
    # Une dimension non comparable doit se declarer non definissable — son poids se
    # redistribue alors sur les autres, ce que le module sait faire — plutot que de
    # produire un rang qui n'a pas de sens. C'est la meme discipline que pour
    # l'autonomie de tresorerie : on ne note pas ce qu'on ne peut pas comparer.
    if len(composantes) < 2:
        return None, None
    return sum(composantes) / len(composantes), "dispersion des marges et de la croissance"


# --------------------------------------------------------------------------- #
# D5 — Dilution de l'actionnaire
# --------------------------------------------------------------------------- #
def d5_dilution(fund, F, reg):
    """Signal = rythme d'emission d'actions nouvelles.

    Reserve doctrinale, a afficher : chez Damodaran, une emission AU PRIX DE MARCHE
    ne transfere aucune valeur. Ce qui appauvrit l'actionnaire, ce sont les options
    et bons emis SOUS la valeur, qu'il valorise et soustrait explicitement. Cette
    dimension mesure donc une realite empirique des micro-capitalisations, pas un
    enseignement de Damodaran."""
    # MEME PIEGE. La serie etait refermee sur ses trous avant d'etre lue par indices
    # voisins ET avant que le nombre d'ANNEES ne serve d'exposant au taux compose :
    # un exercice manquant raccourcissait la duree attribuee a une dilution donnee,
    # et la surestimait donc deux fois.
    actions = list((F or {}).get("shares") or [])
    positif = lambda a: renseigne(a) and a > 0
    voisines = paires_voisines(actions, positif)
    if not voisines:
        return None, None
    recente = (voisines[0][0] / voisines[0][1]) - 1.0
    # L'exposant se compte sur les INTERVALLES parcourus, jamais sur le nombre de
    # valeurs retenues.
    bornes = portee(actions, positif)
    if bornes is not None and bornes[2] >= 3:
        recent, ancien, annees = bornes
        composee = (recent / ancien) ** (1.0 / annees) - 1.0
    else:
        composee = recente
    # Les deux composantes sont fortement redondantes : on retient la PIRE, jamais la
    # somme, qui compterait deux fois la meme emission.
    signal = max(recente, composee)
    if signal <= 0:
        return -math.inf, "aucune dilution : rachats nets ou capital stable"
    return _borne(signal, 0.0, 10.0), "rythme d'emission d'actions nouvelles"


# --------------------------------------------------------------------------- #
# D6 — Liquidite de sortie, RESIDUELLE
# --------------------------------------------------------------------------- #
def d6_liquidite(fund, F, reg, cal=None):
    """Signal = ecart du volume echange a ce que la TAILLE laisse attendre.

    On n'entre NI le volume NI la capitalisation, mais le RESIDU de l'un sur l'autre.
    La part d'illiquidite qu'explique la taille est en effet deja tarifee dans le cout
    des fonds propres, par la prime de taille : l'entrer a nouveau la compterait deux
    fois, et la note ne ferait que reproduire un classement par capitalisation.

    Le residu se calcule ICI et non dans le module de score. Il y etait, et la
    notation remplacait alors le signal APRES que le calibrage eut mesure ses
    quantiles sur la grandeur brute : la table portait sur des `-log10(volume)`, de
    l'ordre de -6 a -9, tandis qu'on y cherchait le rang de residus de l'ordre de
    l'unite. Tout residu depassait donc tout centile, et les 16 000 titres de
    l'univers — Apple comprise, la valeur la plus echangee au monde — ressortaient au
    96e centile d'ILLIQUIDITE. Le calibrage doit mesurer EXACTEMENT la grandeur que la
    notation classe : une seule fonction, un seul chemin."""
    vol, cap = fund.get("volume_dollars_median"), fund.get("market_cap")
    if not vol or vol <= 0 or not cap or cap <= 0:
        return None, None
    droites = (cal or {}).get("liquidite") or {}
    d = droites.get(fund.get("exchange") or "") or droites.get("global")
    if not d:
        return None, None
    attendu = d["a"] + d["b"] * math.log10(cap * 1e9)
    return -(math.log10(vol) - attendu), "volume echange, corrige de la taille"


# --------------------------------------------------------------------------- #
# D7 — Confiance dans la donnee
# --------------------------------------------------------------------------- #
def d7_confiance(fund, F, reg, motifs=None, reparations=None):
    """Signal = opacite. C'est la dimension qui rend l'univers exhaustif, et le SEUL
    endroit du systeme ou l'opacite est comptee — une seule fois.

    Une note de risque qui ignorerait la qualite de ses propres entrees mentirait par
    omission : une societe dont les comptes ont douze ans n'est pas notee, elle est
    devinee. Aucune agence ne mesure cela ; nous le pouvons, parce que nous
    connaissons notre chaine de donnees."""
    points = []
    mois = fund.get("age_des_comptes_mois")
    if mois is not None:
        # Continu, jamais binaire : la queue va jusqu'a des comptes des annees 1990.
        points.append(math.log1p(max(mois - 12, 0) / 12.0))
    profondeur = len([x for x in ((F or {}).get("years") or [])])
    if profondeur:
        points.append(1.0 / profondeur)
    ecart = fund.get("ecart_ebit")
    if ecart is not None:
        # Mode propre a zero : l'ecart nul est le cas NORMAL, il ne doit pas etre
        # penalise par un rang median.
        points.append(min(abs(ecart), 5.0))
    publiees, deduites = fund.get("actions_publiees"), fund.get("shares")
    if publiees and deduites and publiees > 0 and deduites > 0:
        points.append(abs(math.log10(deduites / publiees)))
    # DESACCORD ENTRE LE MARCHE ET LE BILAN. mF International declare 3,66 Md HKD de
    # placements a court terme — 92 % de son actif — pour 34 M HKD de chiffre
    # d'affaires, et cote 108 M HKD. Toutes les identites comptables tiennent, la
    # liasse est coherente avec elle-meme : nous ne pouvons pas trancher avec une
    # seule source.
    # C'est precisement pourquoi ce constat n'est PAS un motif de rejet — nous avons
    # supprime ces seuils, qui confondaient une valorisation extreme avec une donnee
    # fausse. Il est ici MESURE et CLASSE parmi les autres : un ecart de cet ordre
    # entre ce que le bilan affirme et ce que le marche paie signale soit un actif
    # qui n'existe pas, soit un actionnaire qui n'y aura jamais acces. Les deux sont
    # des risques, et aucun ne se lit dans l'upside.
    fp, cap = fund.get("book_equity"), fund.get("market_cap")
    if fp and cap and fp > 0 and cap > 0:
        points.append(max(math.log10(fp / cap), 0.0))
    # POIDS DE SEVERITE RELATIVE — POSES, et l'en-tete du module les passait sous
    # silence en affirmant que « aucun seuil de valeur n'apparait ici ». Ils ne
    # sont ni arithmetiques ni mesures : ils disent seulement qu'un taux de change
    # introuvable est plus grave qu'une devise etrangere connue, et un motif de
    # non-verifiabilite plus grave qu'une reparation reussie. C'est une opinion,
    # defendable, mais une opinion — et elle deplace le signal brut, donc le rang,
    # donc la note. La mesure du 30 juillet 2026 donne d'ailleurs a cette dimension
    # une AUC de 0,496 : sur cette fenetre, elle n'annonce aucun effondrement.
    if fund.get("fx_indisponible"):
        points.append(_D7_FX_INTROUVABLE)
    if (fund.get("financial_currency") or "USD") != "USD":
        points.append(_D7_DEVISE_ETRANGERE)
    if (fund.get("exchange") or "") == "OTC":
        points.append(_D7_GRE_A_GRE)
    points.append(_D7_PAR_MOTIF * len(motifs or []))
    points.append(_D7_PAR_REPARATION * len(reparations or []))
    if not points:
        return None, None
    return sum(points), "opacite et fraicheur des comptes"


# --------------------------------------------------------------------------- #
# D8 — Mur de refinancement a douze mois
# --------------------------------------------------------------------------- #
def d8_refinancement(fund, F, reg):
    """Signal = dette exigible sous douze mois rapportee aux ressources disponibles.

    Rang d'INDUSTRIE : c'est la seule dimension ou ce niveau se justifie. Une
    fonciere refinance en permanence, un producteur de materiaux non ; la comparer a
    l'univers entier n'aurait aucun sens."""
    court = fund.get("short_term_debt")
    if court is None:
        return None, None
    court = abs(court) + abs(fund.get("capital_lease_current") or 0.0)
    if court <= 0:
        return -math.inf, "aucune echeance a douze mois"
    ressources = max(fund.get("cash") or 0.0, 0.0) + max(fund.get("cfo") or 0.0, 0.0)
    if ressources <= 0:
        return 50.0, "echeance a douze mois sans ressource pour y faire face"
    return _borne(court / ressources, 0.0, 50.0), "mur de refinancement a douze mois"


# --------------------------------------------------------------------------- #
# CONSTANTES POSEES DU MODULE, rassemblees et nommees.
#
# L'en-tete affirme que « aucun seuil de valeur n'apparait ici ». C'etait vrai des
# huit dimensions elles-memes — toutes converties en rang contre une table mesuree
# — mais faux de deux endroits : le seuil de bascule vers le regime d'amortissement
# lourd, et les poids de severite relative de la dimension de confiance. Les nommer
# ici ne les rend pas mesurees ; cela les rend VISIBLES, ce qui est la condition
# pour qu'un jour on les mesure ou qu'on les conteste.
# --------------------------------------------------------------------------- #
_SEUIL_AMORTISSEMENT_POSE = 0.40   # part de l'amortissement dans EBIT + amortissement
_D7_FX_INTROUVABLE = 1.0           # aucun taux de change : les montants sont indechiffrables
_D7_PAR_MOTIF = 0.75               # par motif de non-verifiabilite subsistant
_D7_GRE_A_GRE = 0.5                # cotation de gre a gre : obligations de publication moindres
_D7_PAR_REPARATION = 0.5           # par reparation appliquee aux comptes
_D7_DEVISE_ETRANGERE = 0.25        # comptes tenus hors dollar : une conversion de plus


DIMENSIONS = (
    ("d1", "Solvabilité", d1_solvabilite, "univers"),
    ("d2", "Autonomie de trésorerie", d2_autonomie, "univers"),
    ("d3", "Rentabilité de cycle", d3_rentabilite, "secteur"),
    ("d4", "Volatilité d'exploitation", d4_volatilite, "secteur"),
    ("d5", "Dilution", d5_dilution, "univers"),
    ("d6", "Liquidité de sortie", d6_liquidite, "univers"),
    ("d7", "Confiance dans la donnée", d7_confiance, "univers"),
    ("d8", "Mur de refinancement", d8_refinancement, "industrie"),
)


def mesurer(fund, F=None, motifs=None, reparations=None, cal=None):
    """Signaux bruts des huit dimensions. Retourne {cle: (signal, libelle)} ; le
    signal vaut None quand la dimension n'est pas definissable pour ce titre."""
    cal = cal or {}
    reg = regime(fund, F, cal.get("seuil_amortissement"))
    out = {"__regime__": reg}
    for cle, _nom, fonction, _niveau in DIMENSIONS:
        try:
            if cle == "d7":
                out[cle] = fonction(fund, F, reg, motifs, reparations)
            elif cle == "d6":
                # La liquidite a besoin du CALIBRAGE : son signal est un residu de
                # regression, pas une grandeur brute.
                out[cle] = fonction(fund, F, reg, cal)
            else:
                out[cle] = fonction(fund, F, reg)
        except Exception:                              # noqa: BLE001
            out[cle] = (None, None)
    return out


__all__ = ["DIMENSIONS", "mesurer", "regime"]
