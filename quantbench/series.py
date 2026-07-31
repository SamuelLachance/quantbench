"""Lire une serie d'exercices SANS refermer ses trous.

LE DEFAUT QUE CE MODULE EXISTE POUR INTERDIRE. Les series historiques arrivent
alignees sur une meme liste d'exercices : un element par exercice depose, dans un
ordre connu. Les purger de leurs valeurs manquantes — `[v for v in serie if v]` —
retire les trous SANS retirer leur place : la serie se referme sur elle-meme et
deux exercices distants de deux ans deviennent voisins dans la liste.

Tout ce qui se lit ensuite par indices en sort faux :

    serie[i] / serie[i+1]        rapport de deux annees qui ne se suivent pas
    (a / b) ** (1 / (n - 1))     exposant compte sur les VALEURS RETENUES et non
                                 sur les ANNEES parcourues
    len(serie) >= k              profondeur d'historique surestimee

Le biais n'est pas neutre : il RACCOURCIT toujours la duree attribuee a une
variation donnee. Sur une societe en croissance il surestime donc la croissance,
et par elle la valeur ; sur une dilution il surestime le rythme d'emission.

Le motif a ete corrige quatre fois separement — `conversion_en_tresorerie`,
`marge_de_cycle`, la marge normalisee des regulees, la volatilite des capitaux
propres des holdings — et il est reapparu cinq fois ailleurs. Une regle corrigee
au cas par cas finit par diverger : c'est ce qui avait impose `bilan.py`. Elle
est donc ecrite ici, une fois.

CE MODULE NE CONNAIT PAS LE SENS DES SERIES, et c'est voulu : `financials_from_fmp`
les trie PLUS RECENT EN TETE tandis que `revenue_history` est chronologique
croissant. Le sens appartient a l'appelant, qui seul sait ce qu'il lit.
"""
from __future__ import annotations

__all__ = ["renseigne", "paires_voisines", "portee", "serie_continue"]


def renseigne(v) -> bool:
    """Un exercice est renseigne des lors qu'il porte une valeur.

    ZERO EST UNE VALEUR. Un chiffre d'affaires nul, un resultat nul, une
    tresorerie nulle sont des mesures et non des absences — c'est le second
    defaut recurrent de ce depot, et le test par defaut ne doit pas le
    reintroduire. Les appelants qui ont besoin d'une condition plus stricte
    (un logarithme exige un argument strictement positif) la passent
    explicitement.
    """
    return v is not None and v == v          # ecarte aussi les NaN


def paires_voisines(serie, valide=renseigne):
    """Couples `(serie[i], serie[i+1])` dont les DEUX exercices sont renseignes.

    C'est la seule facon licite de tirer un rapport d'annee sur annee d'une serie
    trouee : on n'apparie que des indices reellement adjacents, et un trou fait
    simplement disparaitre les deux couples qui le touchent au lieu de fabriquer
    un couple qui enjambe deux ans.
    """
    s = list(serie or [])
    return [(s[i], s[i + 1]) for i in range(len(s) - 1)
            if valide(s[i]) and valide(s[i + 1])]


def portee(serie, valide=renseigne):
    """`(valeur au premier indice renseigne, valeur au dernier, nombre d'INTERVALLES)`.

    Rend `None` si moins de deux exercices sont renseignes.

    Le nombre d'intervalles se compte sur les INDICES D'ORIGINE. C'est lui, et
    jamais le nombre de valeurs retenues, qui sert d'exposant a un taux compose :
    une serie de six exercices dont un manque parcourt cinq intervalles avec cinq
    valeurs, et prendre la racine quatrieme au lieu de la cinquieme annualise une
    variation qui s'est etalee sur cinq ans.
    """
    idx = [i for i, v in enumerate(serie or []) if valide(v)]
    if len(idx) < 2:
        return None
    s = list(serie)
    return s[idx[0]], s[idx[-1]], idx[-1] - idx[0]


def _annee(cle):
    """Exercice porte par une cle, quelle que soit sa forme.

    Les trois producteurs de series ne designent pas leurs exercices de la meme
    facon : `edgar.annual_series` rend des DATES DE CLOTURE (`"2024-12-31"`),
    `financials_from_fmp` des annees en chaine (`"2024"`), et les jeux de donnees
    SEC des annees entieres. Rendre `None` plutot que de deviner : une cle
    illisible ne doit pas silencieusement devenir l'exercice zero.
    """
    if isinstance(cle, bool):
        return None
    if isinstance(cle, int):
        return cle
    if isinstance(cle, float):
        return int(cle) if cle == cle else None
    try:
        return int(str(cle).strip()[:4])
    except (TypeError, ValueError):
        return None


def serie_continue(par_annee):
    """`(annees, valeurs)` sur un intervalle d'annees SANS trou d'INDICE.

    Une serie batie sur les seules annees PRESENTES — `[m[a] for a in sorted(m)]` —
    porte ses trous de facon INVISIBLE : elle ne contient aucun `None`, et pourtant
    deux elements voisins peuvent etre distants de deux ans. C'est la forme la plus
    dangereuse du defaut, puisque rien dans la liste ne permet de la detecter.

    On reconstruit donc l'intervalle complet, de la plus ancienne a la plus recente
    annee renseignee, en posant `None` sur les annees absentes. Le trou devient
    lisible, et `paires_voisines` comme `portee` savent alors le traiter.

    Rend la serie CHRONOLOGIQUE CROISSANTE — la convention de `revenue_history`.
    Accepte un dictionnaire `{annee: valeur}` ou une suite de couples.
    """
    couples = par_annee.items() if isinstance(par_annee, dict) else par_annee
    m = {}
    for a, v in couples:
        an = _annee(a)
        if an is not None and renseigne(v):
            m[an] = v
    if not m:
        return [], []
    annees = list(range(min(m), max(m) + 1))
    return annees, [m.get(a) for a in annees]
