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
div_ex_date = datetime.datetime(2027, 2, 2)
mkt = Market(100.0, 0.20, 0.03, 0.0, div_ex_date)
mkt_div = Market(100.0, 0.20, 0.03, 3.0, div_ex_date)


# =============== PARAMÈTRES DU MODÈLE ===============
# Date de pricing et nombre de pas (seuil de pruning, 0 = pas de pruning)
mdl = Model(datetime.datetime(2026, 9, 15), 10)
prune_proba = 1e-7


# =============== PARAMÈTRES DE L'OPTION ===============
# maturité, call / put, européenne / américaine, strike
maturity = datetime.datetime(2027, 9, 2)
opt = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL, 101.0)


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

# l'arbre binomial refuse un dividende : il faut l'arbre trinomial (Study passe au trinomial tout seul)
try:
    Tree(mkt_div, mdl, maturity)
except ValueError as error:
    print("binomial avec dividende :", error)

# même vérification avec le dividende (arbre trinomial) : call avec K = 0 vaut S - dividende actualisé
tri_k0 = TrinomialTree(mkt_div, mdl, maturity)
tri_k0.build_tree()
t_ex = (div_ex_date - mdl.price_date).total_seconds() / 86400 / 365
s_minus_div = mkt_div.spot_price - mkt_div.div_amount * math.exp(-mkt_div.int_rate * t_ex)
print("call avec K = 0 et dividende, écart avec S - D actualisé :", tri_k0.price_option(call_k0) - s_minus_div)

# 2. arbre trinomial avec dividende, call américain : exercice juste avant la date ex-div
print("============= Arbre trinomial avec dividende, call américain =============")
amer_call = OptionTrade(maturity, OptionTrade.CALL_LABEL, OptionTrade.AMER_LABEL, 101.0)
tri_tree = TrinomialTree(mkt_div, mdl, maturity)
tri_tree.build_tree()
print(amer_call, "avec un arbre trinomial et un dividende :", tri_tree.price_option(amer_call))
TreeImage(1280, 800, tri_tree, amer_call).save('04_arbre_trinomial_dividende.svg')

# monitoring du même arbre avec un call européen : somme des probas = 1, S moyen = tronc,
# valeur moyenne actualisée de l'option = prix de la racine à chaque date
print("============= Monitoring de l'arbre trinomial, call européen =============")
print("prix :", tri_tree.price_option(opt), " noeuds avec une proba négative :", tri_tree.negative_nodes,
      " proba totale de ces noeuds :", tri_tree.negative_proba_weight())
tri_tree.print_monitoring(1)

# dividende trop grand (50 pour un spot de 100) : sur la date du dividende, la variance demandée
# ne colle plus avec l'espace entre les noeuds et des probas de transition deviennent négatives ;
# le prix n'est alors plus fiable. Le call K = 0 ne le voit pas : les probas, même négatives,
# respectent toujours l'espérance. C'est pour ça qu'on compte ces noeuds.
print("============= Dividende trop grand : probas négatives =============")
big_div = Market(100.0, 0.20, 0.03, 50.0, div_ex_date)
big_tree = TrinomialTree(big_div, mdl, maturity)
big_tree.build_tree()
big_s_minus_div = big_div.spot_price - big_div.div_amount * math.exp(-big_div.int_rate * t_ex)
print("noeuds avec une proba négative :", big_tree.negative_nodes,
      " proba totale de ces noeuds :", round(big_tree.negative_proba_weight(), 6))
print("call avec K = 0, écart avec S - D actualisé :", big_tree.price_option(call_k0) - big_s_minus_div)
big_tree = None


#  =============== ÉTUDES SUR LES ARBRES BINOMIAL ET TRINOMIAL ===============

# 3. toutes les études : marché avec dividende (le marché sans dividende est study.mkt0)
study = Study(mkt_div, mdl, opt)

# convergence des arbres binomial et trinomial vers Black-Scholes, résultats aussi dans des csv
# N = 1 à 100, puis tous les 50 pas jusqu'à 1000 avec N et N + 1 (pair et impair)
print("============= Convergence vers Black-Scholes (graphes 1, 7 et 14) =============")
conv_steps = list(range(1, 101)) + [n + i for n in range(150, 1001, 50) for i in (0, 1)]
study.convergence("binomial", conv_steps, False)
study.convergence("trinomial", conv_steps, True)
study.convergence("trinomial avec pruning", conv_steps, True, prune_proba)
study.save_csv("binomial", 'convergence_binomial.csv')
study.save_csv("trinomial", 'convergence_trinomial.csv')
study.save_csv("trinomial avec pruning", 'convergence_trinomial_pruning.csv')
save(study.convergence_gap_graph(), '01_convergence_ecart.png')
save(study.convergence_price_graph(), '07_convergence_prix.png')

# vitesse de convergence (tunnel de |écart| x N) et temps de calcul (exposant estimé à partir de 100 pas)
print("============= Vitesse de convergence et temps de calcul (graphe 14) =============")
study.print_exponents("binomial", 100, False)
study.print_exponents("trinomial", 100, True)
study.print_exponents("trinomial avec pruning", 100, True)
save(study.run_time_graph(["binomial", "trinomial", "trinomial avec pruning"], 100), '14_temps_calcul.png')

# zoom de 1000 à 1100 pas, tous les pas : on voit la vraie forme de l'écart (pas de fausse oscillation)
# trinomial avec pruning pour aller plus vite : même prix que sans pruning à 1e-8 près (voir plus haut)
print("============= Zoom de 1000 à 1100 pas (graphe 1b) =============")
zoom_steps = list(range(1000, 1101))
study.convergence("binomial zoom", zoom_steps, False)
study.convergence("trinomial zoom", zoom_steps, True, prune_proba)
save(study.convergence_zoom_graph(), '01b_convergence_zoom.png')

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

# effet du dividende avec un taux = 0 : la parité donne call - put = S - D - K
print("============= Effet du dividende sur le call et le put, taux = 0, 100 pas =============")
study.print_dividend_effect(100)

# grecques de l'arbre comparées à Black-Scholes (option de l'étude et call OTM), puis le delta selon N
print("============= Grecques de l'arbre trinomial (graphe 12) =============")
study.print_greeks(100, 120.0)
save(study.delta_graph(list(range(10, 301, 10))), '12_delta.png')

# frontière d'exercice du put américain : sans exercice avant la date ex-dividende
print("============= Frontière d'exercice du put américain, 500 pas (graphe 13) =============")
save(study.exercise_boundary_graph(500, prune_proba, study.option(OptionTrade.PUT_LABEL, OptionTrade.AMER_LABEL)),
     '13_frontiere_exercice.png')

# 10 088 pas avec pruning, comparé à l'Excel du prof (call européen K = 103, 300 jours, sans dividende)
print("============= 10 088 pas avec pruning, comparé à l'Excel du prof =============")
prof_date = datetime.datetime(2026, 3, 3)
prof_call = OptionTrade(prof_date + datetime.timedelta(days=300), OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL,
                        103.0)
big_tree = TrinomialTree(Market(100.0, 0.20, 0.03, 0.0, prof_date), Model(prof_date, 10088, prune_proba),
                         prof_call.mat_date)
big_tree.build_tree()
print("valeur intrinsèque  :", big_tree.price_option(prof_call), " Excel du prof : 6.9985032")
print("Black-Scholes en N-1 :", big_tree.price_option(prof_call, True), " Excel du prof : 6.998418767")
print("noeuds :", big_tree.count_nodes())
big_tree = None


#  =============== COMPLÉMENTS : VUS EN COURS, HORS ÉNONCÉ ===============

# dernier pas : valeur intrinsèque en N ou Black-Scholes sur delta_t en N-1 (cours p. 29-31)
print("============= Dernier pas : VI en N ou Black-Scholes en N-1 (graphes 6 et 11) =============")
save(study.last_step_graph([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]),
     '06_dernier_pas.png')
save(study.last_step_strike_graph(10, 70.0, 130.0, 0.25), '11_dernier_pas_strike.png')

# proba d'être dans la zone d'exercice des options américaines, date par date (cours p. 24)
print("============= Probabilité d'être dans la zone d'exercice, par date, 20 pas (graphe 8) =============")
save(study.exercise_graph(20), '08_proba_exercice.png')
