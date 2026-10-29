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
- `main.py` : point d'entrée, lance toutes les études
- `app.py` : application Streamlit

## Choix de modélisation

- Avec un dividende discret, l'arbre binomial n'est pas utilisé : `Study` passe au trinomial,
  et créer directement un `Tree` binomial avec un dividende lève une erreur.
- Pruning : un noeud atteint avec une proba plus petite que le seuil va seulement vers le noeud
  le plus proche de son espérance (créé s'il n'existe pas), avec une proba de 1.
- Delta et gamma de deux façons : avec les trois noeuds de la date de pricing (l'arbre commence
  un pas avant), ou avec `OneDimDerivative` et un choc de 1 % sur le spot. L'application prend le choc de spot
  avec Black-Scholes au dernier pas, et les trois noeuds avec la valeur intrinsèque (le choc de spot
  y donne un gamma faux : le prix est linéaire entre deux noeuds).
- Vega, vomma et vanna : chocs de 1 % de vol avec `OneDimDerivative`. Les grecques utilisent
  la même méthode du dernier pas que le prix (Black-Scholes par défaut, qui enlève le bruit de la grille).
- Theta : prix demain - prix aujourd'hui ; si le dividende tombe demain, le spot de demain est ex-dividende.
- Les probas de transition négatives sont comptées pendant la construction (avertissement).
- Sur les noeuds où le forward est plus petit que le dividende, le dividende n'est pas payé.
- Nombre de pas pour une précision donnée : formule de l'erreur du cours, prudente
  (l'écart réel est plus petit que la précision demandée).
