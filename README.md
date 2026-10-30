# Pricer d'options par arbres binomial et trinomial

Projet M2 Dauphine, en Python : calls et puts européens et américains, dividende discret,
pruning, grecques (delta, gamma, vega, vomma, vanna, theta), comparaison avec Black-Scholes,
études de convergence et application Streamlit.

## Lancer le projet

Installer les bibliothèques (une seule fois) :

    python -m pip install -r requirements.txt

Toutes les études (graphes PNG, arbres SVG, fichiers CSV, résultats dans la console) :

    python main.py

L'application interactive :

    python -m streamlit run app.py

## Fichiers

- `market.py`, `model.py`, `optiontrade.py` : données du marché, réglages de l'arbre, option
- `node.py`, `trunknode.py`, `tree.py` : arbre binomial
- `trinomialnode.py`, `trinomialtrunknode.py`, `trinomialtree.py` : arbre trinomial, dividende, pruning
- `blackscholes.py`, `derivatives.py`, `greeks.py` : Black-Scholes, dérivées numériques (première et seconde), grecques
- `study.py` : toutes les études et les graphes
- `treeimage.py`, `vectorimage.py` : images SVG des arbres
- `main.py` : point d'entrée, lance toutes les études demandées, puis une section « Compléments »
  (vus en cours, hors énoncé) : dernier pas en N-1 (graphes 6 et 11) et zone d'exercice (graphe 8)
- `app.py` : application Streamlit : prix, tableau des grecques, image de l'arbre (30 pas ou moins),
  frontière d'exercice d'une américaine, et les graphes d'études (moins de pas que `main.py`,
  calculés quand on clique sur le bouton)

## Choix de modélisation

- Avec un dividende discret, l'arbre binomial n'est pas utilisé : `Study` passe au trinomial,
  et créer directement un `Tree` binomial avec un dividende lève une erreur.
- Pruning : un noeud atteint avec une proba plus petite que le seuil va seulement vers le noeud
  le plus proche de son espérance (créé s'il n'existe pas), avec une proba de 1.
  Une américaine peut y être exercée comme sur les autres noeuds (compté dans la frontière d'exercice).
- Delta et gamma de deux façons : avec les trois noeuds de la date de pricing (l'arbre commence
  un pas avant), ou avec `OneDimDerivative` et un choc de 1 % sur le spot. L'application prend le choc de spot
  avec Black-Scholes au dernier pas, et les trois noeuds avec la valeur intrinsèque (le choc de spot
  y donne un gamma faux : le prix est linéaire entre deux noeuds).
- Vega, vomma et vanna : chocs de 1 % de vol avec `OneDimDerivative`. Les grecques utilisent
  la même méthode du dernier pas que le prix (Black-Scholes par défaut, qui enlève le bruit de la grille).
- Theta : prix demain - prix aujourd'hui, même vol et même spot ; si la date ex-dividende est demain,
  le spot de demain est spot - dividende (sinon le theta compterait la chute du dividende, qui n'est
  pas un effet du temps).
- Date ex-dividende : le dividende compte s'il tombe après la date de pricing (exclue) et au plus tard
  à la maturité (incluse). Le jour de la date de pricing, il est déjà détaché ; le jour de la maturité,
  le payoff est calculé sur le prix ex-dividende.
- Formule de l'erreur du cours (pointillés du graphe 1) : faite pour l'option ATM forward ; notre option
  (K = 101, forward environ 103) en est proche, et la formule majore bien l'écart observé.
- Convergence : zoom de 1000 à 1100 pas, tous les pas (graphe 1b), avec le trinomial avec pruning
  pour gagner du temps (même prix à 1e-8 près).
- Probas de transition négatives : le cours demande des probas positives, mais dit aussi que p_d
  n'est pas toujours entre 0 et 1. Cela arrive quand le dividende est grand devant le prix de l'action :
  sur la date du dividende, la variance demandée ne colle plus avec l'espace entre les noeuds.
  Le test du call K = 0 ne le détecte pas (les probas, même négatives, respectent l'espérance).
  Le code compte donc ces noeuds et la somme de leurs
  probas d'atteinte, affiche un avertissement (console et application) et donne leur poids à côté
  du prix : le prix n'est plus fiable si ce poids n'est pas négligeable. Avec nos paramètres
  (dividende de 3), aucun noeud n'a de proba négative ; avec un dividende de 50, oui (voir `main.py`).
- Sur les noeuds où le forward est plus petit que le dividende, le dividende n'est pas payé
  (limite volontaire du projet).
- Probabilité d'exercice par date (graphe 8) : somme des probas des noeuds où l'option est exercée,
  comme dans le cours. C'est la probabilité d'être dans la zone d'exercice à cette date, pas celle
  d'exercer pour la première fois à cette date (un chemin a pu être exercé avant).
