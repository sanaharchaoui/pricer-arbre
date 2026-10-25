import datetime as datetime
import math
from market import Market
from model import Model
from optiontrade import OptionTrade
from tree import Tree
from trinomialtree import TrinomialTree
from blackscholes import BlackScholes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from treeimage import TreeImage
from study import Study


# enregistre un graphe dans un fichier png
def save(fig: Figure, file_name: str):
    fig.savefig(file_name, dpi=150)
    plt.close(fig)
    print("   enregistré :", file_name)


# =============== PARAMÈTRES DU MARCHÉ ===============
# Spot, Volatilité, Taux d'intérêt, Montant du dividende, Date ex-div
div_ex_date = datetime.datetime(2027, 4, 21)
mkt = Market(100.0, 0.30, 0.03, 0.0, div_ex_date)
mkt_div = Market(100.0, 0.30, 0.03, 4.0, div_ex_date)


# =============== PARAMÈTRES DU MODÈLE ===============
# Date de pricing et nombre de pas (seuil de pruning, 0 = pas de pruning)
mdl = Model(datetime.datetime(2026, 9, 1), 300)
prune_proba = 1e-7


# =============== PARAMÈTRES DE L'OPTION ===============
# maturité, call / put, européenne / américaine, strike
maturity = datetime.datetime(2027, 6, 28)
opt = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.AMER_LABEL, 100.0)


#  =============== CONSTRUCTION DES ARBRES BINOMIAL ET TRINOMIAL ===============
# 1. arbre binomial sans dividende, comparé à Black-Scholes
print("============= Arbre binomial sans dividende, comparé à Black-Scholes =============")
tree = Tree(mkt, mdl, maturity)
tree.build_tree()
price = tree.price_option(opt)
print(opt, "avec un arbre binomial de", mdl.steps, "pas :", price)
TreeImage(1280, 800, tree, opt).save('binomial_tree.svg')
bs = BlackScholes(mkt, mdl.price_date, opt)
if bs.definition_ok:
    print("Black-Scholes :", bs.price(), " écart :", price - bs.price())

# vérifications vues en cours : call avec K = 0 vaut le spot, call = put quand K = forward
print("============= Vérifications : K = 0 et K = forward =============")
call_k0 = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL, 0.0)
print("call avec K = 0, écart avec le spot :", tree.price_option(call_k0) - mkt.spot_price)
fwd = mkt.spot_price * math.exp(mkt.int_rate * tree.delta_t * mdl.steps)
call_f = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL, fwd)
put_f = OptionTrade(maturity, OptionTrade.PUT_LABEL, OptionTrade.EURO_LABEL, fwd)
print("K = forward, call - put :", tree.price_option(call_f) - tree.price_option(put_f))

# 2. arbre trinomial avec dividende, call américain : exercice juste avant la date ex-div
print("============= Arbre trinomial avec dividende, call américain =============")
amer_call = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.AMER_LABEL, 100.0)
tri_tree = TrinomialTree(mkt_div, mdl, maturity)
tri_tree.build_tree()
print(amer_call, "avec un arbre trinomial et un dividende :", tri_tree.price_option(amer_call))
TreeImage(1280, 800, tri_tree, amer_call).save('04_arbre_trinomial_dividende.svg')

# monitoring du même arbre avec un call européen : somme des probas = 1, S moyen = tronc,
# valeur moyenne actualisée de l'option = prix de la racine à chaque date
print("============= Monitoring de l'arbre trinomial, call européen =============")
print("prix :", tri_tree.price_option(opt))
tri_tree.print_monitoring(1)


#  =============== ÉTUDES SUR LES ARBRES BINOMIAL ET TRINOMIAL ===============

# 3. toutes les études : marché avec dividende (le marché sans dividende est study.mkt0)
study = Study(mkt_div, mdl, opt)

# convergence des arbres binomial et trinomial vers Black-Scholes, résultats aussi dans des csv
print("============= Convergence vers Black-Scholes (graphes 1 et 7) =============")
study.convergence("binomial", 300, False)
study.convergence("trinomial", 300, True)
study.convergence("trinomial avec pruning", 300, True, prune_proba)
study.save_csv("binomial", 'convergence_binomial.csv')
study.save_csv("trinomial", 'convergence_trinomial.csv')
study.save_csv("trinomial avec pruning", 'convergence_trinomial_pruning.csv')
save(study.convergence_gap_graph(), '01_convergence_ecart.png')
save(study.convergence_price_graph(), '07_convergence_prix.png')

# écart avec Black-Scholes selon le strike : en arches, avec un minimum sur chaque noeud de la maturité
print("============= Écart avec Black-Scholes en fonction du strike, 10 pas (graphe 2) =============")
save(study.strike_gap_graph(10, 80.0, 120.0, 0.1), '02_ecart_strike.png')

# différence entre américaine et européenne selon le taux, sans puis avec dividende
print("============= Américaine - européenne en fonction du taux, 100 pas (graphe 3) =============")
save(study.rate_graph(100, -0.05, 0.05, 0.01), '03_effet_taux.png')

# pruning : nombre de noeuds et temps de calcul, sans puis avec pruning
print("============= Pruning : effet sur le temps de calcul (graphe 5) =============")
study.print_pruning_effect(1000, prune_proba)
save(study.pruning_graph([10, 20, 50, 100, 200, 300, 500, 700, 1000], prune_proba), '05_pruning_temps.png')

# dernier pas : valeur intrinsèque en N ou Black-Scholes sur delta_t en N-1
print("============= Dernier pas : VI en N ou Black-Scholes en N-1 (graphes 6 et 11) =============")
study.print_last_step(100)
save(study.last_step_graph([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]),
     '06_dernier_pas.png')
save(study.last_step_strike_graph(10, 70.0, 130.0, 0.25), '11_dernier_pas_strike.png')

# probabilité d'exercice des options américaines, date par date
print("============= Probabilité d'exercice par date, 20 pas (graphe 8) =============")
save(study.exercise_graph(20), '08_proba_exercice.png')

# prix des options quand la date ex-dividende bouge, tous les 7 jours
print("============= Effet de la date ex-dividende, 100 pas (graphe 9) =============")
save(study.ex_date_graph(100, 7), '09_date_ex_dividende.png')

# tronc de l'arbre trinomial : chute du prix à la date ex-dividende
print("============= Tronc de l'arbre trinomial avec dividende, 100 pas (graphe 10) =============")
save(study.trunk_graph(100), '10_tronc_dividende.png')

# effet du dividende avec un taux = 0 : la parité donne call - put = S - D - K
print("============= Effet du dividende sur le call et le put, taux = 0, 100 pas =============")
study.print_dividend_effect(100)

# put avec dividende comparé à deux approximations sans dividende (spot - D et strike + D)
print("============= Put avec dividende de 4, comparé à des puts sans dividende, 500 pas =============")
study.print_dividend_approximation(500, 4.0, mdl.price_date + datetime.timedelta(days=1))
study.print_dividend_approximation(500, 4.0, datetime.datetime(2027, 3, 1))
study.print_dividend_approximation(500, 4.0, maturity - datetime.timedelta(days=1))

# même option avec le binomial, le trinomial et le trinomial avec l'espace du binomial
print("============= Arbres sans dividende, 100 pas =============")
study.print_tree_comparison(mdl.steps)

# delta, gamma et vega de l'arbre comparés à Black-Scholes, puis le delta selon le nombre de pas
print("============= Grecques de l'arbre trinomial (graphe 12) =============")
study.print_greeks(mdl.steps)
save(study.delta_graph(list(range(10, 301, 10))), '12_delta.png')

# frontière d'exercice du put américain : sans exercice avant la date ex-dividende
print("============= Frontière d'exercice du put américain, 500 pas (graphe 13) =============")
save(study.exercise_boundary_graph(500, prune_proba), '13_frontiere_exercice.png')

# monitoring avec pruning et dividende, 1000 pas : la somme des probas reste 1 à chaque date
print("============= Monitoring avec pruning et dividende, 1000 pas =============")
pruned_tree = TrinomialTree(mkt_div, Model(mdl.price_date, 1000, prune_proba), maturity)
pruned_tree.build_tree()
euro_put = OptionTrade(maturity, OptionTrade.PUT_LABEL, OptionTrade.EURO_LABEL, 100.0)
print("prix :", pruned_tree.price_option(euro_put))
pruned_tree.print_monitoring(250)
