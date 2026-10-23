# Pricer d'options par arbres binomial et trinomial

Projet M2 Dauphine, en Python : calls et puts européens et américains, dividende discret,
pruning, grecques, comparaison avec Black-Scholes, études de convergence et application Streamlit.

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
- `blackscholes.py`, `derivatives.py`, `greeks.py` : Black-Scholes, dérivée numérique, grecques
- `study.py` : toutes les études et les graphes
- `treeimage.py`, `vectorimage.py` : images SVG des arbres
- `main.py` : point d'entrée, lance toutes les études
- `app.py` : application Streamlit
