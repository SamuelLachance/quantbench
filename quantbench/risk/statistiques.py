"""Les mesures qui servent a juger la note de risque. Une seule ecriture.

Deux scripts jugent cette note : `valider_les_notes.py`, qui la reconstruit a une
date passee et regarde ce qui est arrive depuis, et `mesurer_les_notes.py`, qui
confrontera deux archives mensuelles des qu'un an les separera. Ils avaient chacun
leur aire sous la courbe. Elles s'accordaient — verifie, et a deux dix-millionemes
de milliardieme de sklearn — mais l'une rendait l'ecart-type de la mesure et
l'autre non, si bien que le tableau publie sur le site affichait des AUC sans
jamais dire a quelle precision. Un chiffre sans son incertitude ne se lit pas :
0,545 et 0,491 se distinguent si l'ecart-type vaut 0,013, pas s'il vaut 0,05.

Aucune dependance a scipy ni a scikit-learn : ces fonctions tournent dans des jobs
d'integration continue ou l'on veut installer le moins possible, et elles sont
assez courtes pour etre lues en entier.
"""
from __future__ import annotations

import bisect
import math

__all__ = ["aire_sous_la_courbe", "wilson", "test_deux_proportions",
           "spearman", "mediane"]


def aire_sous_la_courbe(paires):
    """(AUC, ecart-type). `paires` = [(score, effondre), ...].

    AUC = probabilite qu'une societe qui s'est effondree porte un score PIRE
    qu'une societe epargnee tiree au hasard. 0,50 signifie exactement aucune
    information, et c'est la seule lecture qui compte : une dimension a 0,50 pese
    dans la note sans rien y apporter.

    Calcul par rangs (statistique de Mann-Whitney) : exact, sans hypothese de
    distribution, et les ex aequo comptent pour une demie — sans quoi une dimension
    a valeurs discretes serait creditee d'un pouvoir qu'elle n'a pas.

    L'ecart-type suit Hanley et McNeil (1982). Il n'est pas decoratif : il decide
    si un ecart a 0,50 est une information ou du bruit d'echantillonnage.
    """
    pos = [s for s, y in paires if y]
    neg = [s for s, y in paires if not y]
    if not pos or not neg:
        return None, None

    neg_tries = sorted(neg)
    total = 0.0
    for s in pos:
        inf = bisect.bisect_left(neg_tries, s)
        egaux = bisect.bisect_right(neg_tries, s) - inf
        total += inf + 0.5 * egaux
    n1, n0 = len(pos), len(neg)
    auc = total / (n1 * n0)

    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n1 - 1) * (q1 - auc ** 2)
           + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return auc, math.sqrt(max(var, 0.0))


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson pour une proportion.

    Prefere a l'intervalle normal : avec dix observations et zero effondrement,
    l'intervalle normal donne [0 %, 0 %], ce qui affirmerait une certitude que dix
    observations ne peuvent pas porter.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - e), min(1.0, c + e))


def test_deux_proportions(k1, n1, k2, n2):
    """Test z sur la DIFFERENCE de deux proportions. Renvoie (z, p bilateral).

    Il remplace une comparaison d'intervalles de confiance, qui avait donne un
    verdict faux : les intervalles de Wilson de A [15,3 % ; 32,1 %] et de F
    [32,0 % ; 52,4 %] se chevauchaient d'un dixieme de point, d'ou la conclusion
    "separation non etablie" — alors que le test direct donne p = 0,006. Le
    chevauchement de deux intervalles est un critere CONSERVATEUR : son absence
    prouve la difference, sa presence ne prouve rien.
    """
    if n1 == 0 or n2 == 0:
        return (None, None)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (None, None)
    z = (p2 - p1) / se
    return (z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def spearman(paires):
    """Correlation de rang entre deux series. None si moins de trois points."""
    if len(paires) < 3:
        return None
    xs = [p[0] for p in paires]
    ys = [p[1] for p in paires]

    def rangs(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            moy = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = moy
            i = j + 1
        return r

    rx, ry = rangs(xs), rangs(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def mediane(xs):
    s = sorted(xs)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
