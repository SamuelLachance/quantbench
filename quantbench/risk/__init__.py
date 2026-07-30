"""Note de risque actionnaire — de A+ a F, pour chaque societe valorisee.

La valorisation repond a "combien vaut cette societe". La note repond a une question
DIFFERENTE : "quelle est la probabilite de perdre durablement sa mise". Un titre peut
etre tres decote et tres dangereux ; c'est meme la configuration la plus frequente.

Trois principes structurent le systeme.

  LE REGIME AVANT LA DIMENSION. Une societe n'est pas mesuree par la meme regle selon
  la nature de son bilan : les interets sont un cout de FINANCEMENT pour un
  industriel et une MATIERE PREMIERE pour une banque. Le regime se determine sur une
  mesure structurelle, jamais sur un libelle sectoriel.

  TOUT ENTRE EN RANG, JAMAIS EN VALEUR BRUTE. Un rang percentile supprime les queues
  non bornees et, surtout, il supprime les seuils : aucune constante de ce module ne
  dit ce qu'est une "bonne" couverture d'interets.

  LE RANG SE LIT DANS UNE TABLE GELEE, PAS DANS L'UNIVERS DU JOUR. Sans cela la note
  serait un simple classement : dans une crise ou toutes les couvertures s'effondrent
  ensemble, personne ne serait degrade. La table est mesuree une fois sur une
  population de reference datee, versionnee, et le build quotidien ne fait que la
  lire — ce qui rend au passage les cinq shards independants.
"""

from .dimensions import DIMENSIONS, mesurer, regime
from .score import GRADES, noter

__all__ = ["DIMENSIONS", "GRADES", "mesurer", "noter", "regime"]
