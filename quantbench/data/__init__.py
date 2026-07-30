"""Connecteurs de donnees : FMP, SEC EDGAR, marche.

L'import de `build_dcf_inputs` est PARESSEUX. Il etait immediat, si bien que le simple
`from quantbench.data import market` — un module hors ligne — declenchait la chaine
`build` -> `edgar` -> `requests`. La suite d'invariants, pourtant concue pour ne
toucher aucun reseau, ne pouvait donc pas s'importer sans client HTTP installe :
l'integration continue echouait a la COLLECTE des tests, sans qu'aucune regression ne
soit en cause.

Un module hors ligne ne doit dependre que de modules hors ligne.
"""

__all__ = ["build_dcf_inputs"]


def __getattr__(name):                          # PEP 562
    if name == "build_dcf_inputs":
        from .build import build_dcf_inputs
        return build_dcf_inputs
    raise AttributeError(f"le module {__name__!r} n'expose pas {name!r}")
