from __future__ import annotations
import datetime as datetime
import gc
import math
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from market import Market
from model import Model
from optiontrade import OptionTrade
from node import Node
from tree import Tree
from trinomialtree import TrinomialTree
from trinomialtrunknode import TrinomialTrunkNode
from blackscholes import BlackScholes
from greeks import Greeks

CALL: str = OptionTrade.CALL_LABEL
PUT: str = OptionTrade.PUT_LABEL
EURO: str = OptionTrade.EURO_LABEL
AMER: str = OptionTrade.AMER_LABEL


# valeurs de x_min à x_max, avec un pas de x_jump
def values_between(x_min: float, x_max: float, x_jump: float) -> list:
    values: list = []
    x: float = x_min
    while x <= x_max + 1e-9:
        values.append(x)
        x = x + x_jump
    return values


# toutes les études du projet : chaque méthode affiche ses résultats dans la console
# ou renvoie un graphe (figure matplotlib) que main.py enregistre
class Study(object):

    # les quatre méthodes du dernier pas : nom -> (arbre trinomial ?, Black-Scholes sur le dernier pas ?)
    LAST_STEP_METHODS: dict = {"Bin avec VI": (False, False), "Trin avec VI": (True, False),
                               "Bin avec option": (False, True), "Trin avec option": (True, True)}

    # mkt contient le dividende, mkt0 est le même marché sans dividende
    def __init__(self, mkt: Market, mdl: Model, opt: OptionTrade):
        self.mkt: Market = mkt
        self.mkt0: Market = self.market(div=0.0)
        self.mdl: Model = mdl
        self.opt: OptionTrade = opt
        # résultats des convergences : nom -> {"pas": [...], "prix": [...], ...}
        self.conv: dict = {}

    # ==================== outils ====================

    # même marché avec un autre taux, un autre dividende ou une autre date ex-div (None : on garde)
    def market(self, rate: float = None, div: float = None, ex_date: datetime.datetime = None) -> Market:
        m: Market = self.mkt
        return Market(m.spot_price, m.volatility, m.int_rate if rate is None else rate,
                      m.div_amount if div is None else div, m.div_ex_date if ex_date is None else ex_date)

    # option de l'étude avec un autre type, un autre exercice ou un autre strike (None : on garde)
    def option(self, call_put: str = None, ex: str = None, strike: float = None) -> OptionTrade:
        o: OptionTrade = self.opt
        return OptionTrade(o.mat_date, o.opt_type if call_put is None else call_put,
                           o.exercice if ex is None else ex, o.strike if strike is None else strike)

    # arbre binomial ou trinomial à n pas, déjà construit
    def tree(self, n: int, trinomial: bool, mkt: Market, prune: float = 0.0) -> Tree:
        mdl: Model = Model(self.mdl.price_date, n, prune)
        if trinomial:
            tree: Tree = TrinomialTree(mkt, mdl, self.opt.mat_date)
        else:
            tree = Tree(mkt, mdl, self.opt.mat_date)
        tree.build_tree()
        return tree

    # prix Black-Scholes, marché sans dividende
    def bs_price(self, opt: OptionTrade) -> float:
        return BlackScholes(self.mkt0, self.mdl.price_date, opt).price()

    # graphe avec une courbe par nom : curves = nom -> (liste des x, liste des y)
    def figure(self, title: str, x_label: str, curves: dict,
               x_log: bool = False, y_log: bool = False) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in curves:
            ax.plot(curves[name][0], curves[name][1], label=name)
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.set(title=title, xlabel=x_label)
        ax.legend()
        fig.tight_layout()
        return fig

    # ==================== convergence et pruning (graphes 1, 5 et 7) ====================

    # prix, écart x pas et temps de calcul pour n = 1 ... n_max, gardés pour les graphes
    def convergence(self, name: str, n_max: int, trinomial: bool, prune: float = 0.0):
        bs: float = self.bs_price(self.opt)
        res: dict = {"pas": [], "prix": [], "ecart x pas": [], "temps": []}
        for n in range(1, n_max + 1):
            gc.collect()  # on libère les vieux arbres, sinon ça compte dans le temps
            t0: float = time.perf_counter()
            price: float = self.tree(n, trinomial, self.mkt0, prune).price_option(self.opt)
            res["temps"].append(time.perf_counter() - t0)
            res["pas"].append(n)
            res["prix"].append(price)
            res["ecart x pas"].append((price - bs) * n)
        self.conv[name] = res
        print("  ", name, ": prix à", n_max, "pas", price, " Black-Scholes", bs)

    # résultats d'une convergence dans un fichier csv
    def save_csv(self, name: str, file_name: str):
        res: dict = self.conv[name]
        with open(file_name, "w") as f:
            f.write(",".join(res) + "\n")
            for i in range(len(res["pas"])):
                f.write(",".join([str(res[key][i]) for key in res]) + "\n")

    # une courbe par convergence : "key" en fonction du nombre de pas
    def conv_curves(self, names: list, key: str) -> dict:
        return {name: (self.conv[name]["pas"], self.conv[name][key]) for name in names}

    # graphe 1 : (prix - Black-Scholes) x pas, binomial et trinomial
    def convergence_gap_graph(self) -> Figure:
        return self.figure("(Prix arbre - Black-Scholes) x nombre de pas", "Nombre de pas",
                           self.conv_curves(["binomial", "trinomial"], "ecart x pas"))

    # graphe 7 : prix en fonction du nombre de pas, avec le prix Black-Scholes
    def convergence_price_graph(self) -> Figure:
        curves: dict = self.conv_curves(["binomial", "trinomial"], "prix")
        steps: list = self.conv["binomial"]["pas"]
        curves["Black-Scholes"] = (steps, [self.bs_price(self.opt)] * len(steps))
        return self.figure("Prix de l'option en fonction du nombre de pas", "Nombre de pas", curves)

    # graphe 5 : temps de calcul du trinomial avec dividende, sans et avec pruning, échelles log
    def pruning_graph(self, n_list: list, prune: float) -> Figure:
        curves: dict = {}
        for p in (0.0, prune):
            curves["pruning " + str(p)] = (n_list, [self.run_time(n, p) for n in n_list])
        return self.figure("Temps de calcul en secondes (construction + pricing)", "Nombre de pas",
                           curves, True, True)

    # temps de calcul (construction + pricing) d'un arbre trinomial à n pas, marché avec dividende
    def run_time(self, n: int, prune: float) -> float:
        gc.collect()  # on libère les vieux arbres, sinon ça compte dans le temps
        t0: float = time.perf_counter()
        self.tree(n, True, self.mkt, prune).price_option(self.opt)
        return time.perf_counter() - t0

    # effet du pruning sur un put américain à n pas : prix, nombre de noeuds et temps (console)
    def print_pruning_effect(self, n: int, prune: float):
        put: OptionTrade = self.option(PUT, AMER)
        for p in (0.0, prune):
            gc.collect()
            t0: float = time.perf_counter()
            tree: Tree = self.tree(n, True, self.mkt, p)
            price: float = tree.price_option(put)
            print("   pruning", p, ": prix", round(price, 8), " noeuds", tree.count_nodes(),
                  " temps", round(time.perf_counter() - t0, 2), "s")

    # ==================== strike (graphes 2 et 11) ====================

    # graphe 2 : écart avec Black-Scholes en fonction du strike, arbre binomial à n pas
    # (pointillés : noeuds de la maturité)
    def strike_gap_graph(self, n: int, k_min: float, k_max: float, k_jump: float) -> Figure:
        tree: Tree = self.tree(n, False, self.mkt0)
        strikes: list = values_between(k_min, k_max, k_jump)
        gaps: list = [tree.price_option(self.option(strike=k)) - self.bs_price(self.option(strike=k))
                      for k in strikes]
        fig: Figure = self.figure("Prix arbre - Black-Scholes en fonction du strike (" + str(n) + " pas)",
                                  "Strike", {"arbre binomial": (strikes, gaps)})
        for s in self.last_column(tree):
            if k_min <= s <= k_max:
                fig.axes[0].axvline(s, color="grey", linestyle=":")
        return fig

    # prix du sous-jacent sur les noeuds de la maturité, du haut vers le bas
    def last_column(self, tree: Tree) -> list:
        n: Node = tree.last_trunk
        while n.unode is not None:
            n = n.unode
        prices: list = []
        while n is not None:
            prices.append(n.under_price)
            n = n.dnode
        return prices

    # graphe 11 : dernier pas, écart avec Black-Scholes en fonction du strike, arbres à n pas
    def last_step_strike_graph(self, n: int, k_min: float, k_max: float, k_jump: float) -> Figure:
        strikes: list = values_between(k_min, k_max, k_jump)
        curves: dict = {}
        for name in Study.LAST_STEP_METHODS:
            trinomial, last_step_bs = Study.LAST_STEP_METHODS[name]
            tree: Tree = self.tree(n, trinomial, self.mkt0)
            curves[name] = (strikes, [tree.price_option(self.option(strike=k), last_step_bs)
                                      - self.bs_price(self.option(strike=k)) for k in strikes])
        return self.figure("Dernier pas : prix arbre - Black-Scholes (" + str(n) + " pas)", "Strike", curves)

    # ==================== dernier pas (graphe 6) ====================

    # graphe 6 : dernier pas, (prix - Black-Scholes) x pas pour les quatre méthodes
    def last_step_graph(self, n_list: list) -> Figure:
        bs: float = self.bs_price(self.opt)
        curves: dict = {}
        for name in Study.LAST_STEP_METHODS:
            trinomial, last_step_bs = Study.LAST_STEP_METHODS[name]
            gaps: list = [(self.tree(n, trinomial, self.mkt0).price_option(self.opt, last_step_bs) - bs) * n
                          for n in n_list]
            curves[name] = (n_list, gaps)
            print("  ", name, ": écart x pas à", n_list[-1], "pas =", round(gaps[-1], 6))
        return self.figure("Dernier pas : (prix arbre - Black-Scholes) x nombre de pas", "Nombre de pas",
                           curves, True)

    # dernier pas avec un arbre trinomial à n pas : VI en N ou Black-Scholes en N-1 (console)
    def print_last_step(self, n: int):
        tree: Tree = self.tree(n, True, self.mkt0)
        for call_put in (CALL, PUT):
            for ex in (EURO, AMER):
                opt: OptionTrade = self.option(call_put, ex)
                print("  ", opt, ": VI", round(tree.price_option(opt), 6),
                      " option", round(tree.price_option(opt, True), 6))
        print("   Black-Scholes call :", self.bs_price(self.option(CALL, EURO)))

    # ==================== exercice américain (graphes 3 et 8) ====================

    # graphe 3 : américaine - européenne en fonction du taux, call et put, sans et avec dividende
    def rate_graph(self, n: int, r_min: float, r_max: float, r_jump: float) -> Figure:
        rates: list = values_between(r_min, r_max, r_jump)
        curves: dict = {}
        for div in (0.0, self.mkt.div_amount):
            for call_put in (CALL, PUT):
                diffs: list = [self.amer_minus_euro(n, self.market(r, div), call_put) for r in rates]
                curves[call_put + ", dividende " + str(div)] = (rates, diffs)
                print("  ", call_put, "dividende", div, ":", [round(d, 4) for d in diffs])
        fig: Figure = self.figure("Américaine - européenne en fonction du taux", "Taux d'intérêt", curves)
        fig.axes[0].axvline(0.0, color="black", linewidth=0.5)
        return fig

    # prix américain - prix européen sur le même arbre trinomial
    def amer_minus_euro(self, n: int, mkt: Market, call_put: str) -> float:
        tree: Tree = self.tree(n, True, mkt)
        return tree.price_option(self.option(call_put, AMER)) - tree.price_option(self.option(call_put, EURO))

    # graphe 8 : probabilité d'exercice par date des options américaines, arbre trinomial à n pas
    def exercise_graph(self, n: int) -> Figure:
        curves: dict = {}
        for call_put, mkt in ((PUT, self.mkt0), (PUT, self.mkt), (CALL, self.mkt)):
            name: str = call_put + " américain, dividende " + str(mkt.div_amount)
            curves[name] = self.exercise_probas(self.tree(n, True, mkt), self.option(call_put, AMER))
        return self.figure("Probabilité d'exercice par date (" + str(n) + " pas)", "Date", curves)

    # dates du tronc et proba d'exercice à chaque date, après le pricing de opt
    def exercise_probas(self, tree: TrinomialTree, opt: OptionTrade) -> tuple:
        tree.price_option(opt)
        dates: list = []
        probas: list = []
        trunk: TrinomialTrunkNode = tree.root
        while trunk is not None:
            dates.append(trunk.column_date)
            probas.append(tree.column_exercise_proba(trunk, opt))
            trunk = trunk.next_mid
        return dates, probas

    # ==================== dividende (graphes 9 et 10) ====================

    # graphe 9 : prix en fonction de la date ex-div (tous les day_jump jours), arbre trinomial à n pas
    def ex_date_graph(self, n: int, day_jump: int) -> Figure:
        dates: list = []
        d: datetime.datetime = self.mdl.price_date + datetime.timedelta(days=day_jump)
        while d <= self.opt.mat_date:
            dates.append(d)
            d = d + datetime.timedelta(days=day_jump)
        trees: list = [self.tree(n, True, self.market(ex_date=d)) for d in dates]
        curves: dict = {}
        for call_put in (CALL, PUT):
            for ex in (EURO, AMER):
                opt: OptionTrade = self.option(call_put, ex)
                curves[call_put + " " + ex] = (dates, [t.price_option(opt) for t in trees])
        return self.figure("Prix en fonction de la date ex-dividende", "Date ex-dividende", curves)

    # graphe 10 : tronc de l'arbre trinomial à n pas, la chute du dividende à la date ex-div
    def trunk_graph(self, n: int) -> Figure:
        trunk: TrinomialTrunkNode = self.tree(n, True, self.mkt).root
        dates: list = []
        prices: list = []
        while trunk is not None:
            dates.append(trunk.column_date)
            prices.append(trunk.under_price)
            trunk = trunk.next_mid
        fig: Figure = self.figure("Tronc de l'arbre trinomial (prix forward)", "Date",
                                  {"tronc": (dates, prices)})
        fig.axes[0].axvline(self.mkt.div_ex_date, color="red", linestyle="--", label="date ex-dividende")
        fig.axes[0].legend()
        return fig

    # effet du dividende avec un taux = 0 : call et put européens sans et avec dividende (console)
    def print_dividend_effect(self, n: int):
        for div in (0.0, self.mkt.div_amount):
            tree: Tree = self.tree(n, True, self.market(0.0, div))
            call: float = tree.price_option(self.option(CALL, EURO))
            put: float = tree.price_option(self.option(PUT, EURO))
            print("   dividende", div, ": call", round(call, 6), " put", round(put, 6),
                  " call - put", round(call - put, 6))

    # put européen avec un dividende div payé à ex_date, comparé à deux puts sans dividende (BS) :
    # spot - dividende actualisé (exact pour un dividende demain)
    # strike + dividende capitalisé (exact pour un dividende juste avant la maturité)
    def print_dividend_approximation(self, n: int, div: float, ex_date: datetime.datetime):
        r: float = self.mkt.int_rate
        t_ex: float = (ex_date - self.mdl.price_date).days / 365
        t_mat: float = (self.opt.mat_date - self.mdl.price_date).days / 365
        put: OptionTrade = self.option(PUT, EURO)
        tree_price: float = self.tree(n, True, self.market(div=div, ex_date=ex_date)).price_option(put)
        spot: float = self.mkt.spot_price - div * math.exp(-r * t_ex)
        mkt_spot: Market = Market(spot, self.mkt.volatility, r, 0.0, ex_date)
        spot_price: float = BlackScholes(mkt_spot, self.mdl.price_date, put).price()
        strike: float = put.strike + div * math.exp(r * (t_mat - t_ex))
        strike_price: float = self.bs_price(self.option(PUT, EURO, strike))
        print("   ex-div", ex_date.date(), ": arbre", round(tree_price, 6), " spot - dividende",
              round(spot_price, 6), " strike + dividende", round(strike_price, 6))

    # ==================== arbres et grecques (graphe 12) ====================

    # arbres sans dividende à n pas : binomial, trinomial, trinomial avec l'espace du binomial (console)
    def print_tree_comparison(self, n: int):
        mdl: Model = Model(self.mdl.price_date, n)
        tri_bin: TrinomialTree = TrinomialTree(self.mkt0, mdl, self.opt.mat_date, True)
        tri_bin.build_tree()
        print("   binomial                    :", self.tree(n, False, self.mkt0).price_option(self.opt))
        print("   trinomial                   :", self.tree(n, True, self.mkt0).price_option(self.opt))
        print("   trinomial, espace binomial  :", tri_bin.price_option(self.opt))
        print("   Black-Scholes               :", self.bs_price(self.opt))

    # graphe 12 : delta de l'arbre trinomial en fonction du nombre de pas, comparé à Black-Scholes
    def delta_graph(self, n_list: list) -> Figure:
        deltas: list = [Greeks(self.mkt0, Model(self.mdl.price_date, n), self.opt).delta() for n in n_list]
        bs: float = BlackScholes(self.mkt0, self.mdl.price_date, self.opt).delta()
        return self.figure("Delta en fonction du nombre de pas", "Nombre de pas",
                           {"arbre trinomial": (n_list, deltas),
                            "Black-Scholes": (n_list, [bs] * len(n_list))})

    # delta, gamma et vega de l'arbre trinomial à n pas, comparés à Black-Scholes (console)
    def print_greeks(self, n: int):
        g: Greeks = Greeks(self.mkt0, Model(self.mdl.price_date, n), self.opt)
        bs: BlackScholes = BlackScholes(self.mkt0, self.mdl.price_date, self.opt)
        print("   arbre", n, "pas : delta", round(g.delta(), 6), " gamma", round(g.gamma(), 6),
              " vega (1 %)", round(g.vega() * 0.01, 6))
        print("   Black-Scholes  : delta", round(bs.delta(), 6), " gamma", round(bs.gamma(), 6),
              " vega (1 %)", round(bs.vega() * 0.01, 6))

    # ==================== frontière d'exercice (graphe 13) ====================

    # graphe 13 : frontière d'exercice du put américain, sans et avec dividende, arbre trinomial à n pas
    # au-dessus de la courbe on garde le put, en dessous on l'exerce
    def exercise_boundary_graph(self, n: int, prune: float) -> Figure:
        curves: dict = {}
        for mkt in (self.mkt0, self.mkt):
            name: str = "put américain, dividende " + str(mkt.div_amount)
            curves[name] = self.boundary(self.tree(n, True, mkt, prune), self.option(PUT, AMER))
            print("  ", name, ": premier exercice anticipé le", curves[name][0][0].date())
        fig: Figure = self.figure("Frontière d'exercice du put américain (" + str(n) + " pas)", "Date",
                                  curves)
        fig.axes[0].axvline(self.mkt.div_ex_date, color="grey", linestyle="--", label="date ex-dividende")
        fig.axes[0].axhline(self.opt.strike, color="black", linestyle=":", label="strike")
        fig.axes[0].legend()
        return fig

    # prix critique S* à chaque date : le prix du noeud le plus haut où l'option est exercée
    # (les dates sans aucun exercice ne sont pas gardées)
    def boundary(self, tree: TrinomialTree, opt: OptionTrade) -> tuple:
        tree.price_option(opt)
        dates: list = []
        s_star: list = []
        trunk: TrinomialTrunkNode = tree.root
        while trunk is not None:
            price: float = self.highest_exercised(trunk, opt)
            if price is not None:
                dates.append(trunk.column_date)
                s_star.append(price)
            trunk = trunk.next_mid
        return dates, s_star

    # on part du haut de la colonne et on descend jusqu'au premier noeud exercé (None si aucun)
    def highest_exercised(self, trunk: TrinomialTrunkNode, opt: OptionTrade) -> float:
        n: Node = trunk
        while n.unode is not None:
            n = n.unode
        while n is not None and not n.is_exercised(opt):
            n = n.dnode
        if n is None:
            return None
        return n.under_price
