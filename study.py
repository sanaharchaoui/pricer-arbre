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
    # le binomial ne gère pas le dividende : s'il y en a un, on passe au trinomial
    def tree(self, n: int, trinomial: bool, mkt: Market, prune: float = 0.0) -> Tree:
        mdl: Model = Model(self.mdl.price_date, n, prune)
        if trinomial or mkt.has_dividend(self.mdl.price_date, self.opt.mat_date):
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

    # ==================== convergence et pruning (graphes 1, 5, 7 et 14) ====================

    # prix, écart x pas et temps de calcul pour chaque nombre de pas de n_list, gardés pour les graphes
    def convergence(self, name: str, n_list: list, trinomial: bool, prune: float = 0.0):
        bs: float = self.bs_price(self.opt)
        res: dict = {"pas": [], "prix": [], "ecart x pas": [], "temps": []}
        for n in n_list:
            gc.collect()  # on libère les vieux arbres, sinon ça compte dans le temps
            t0: float = time.perf_counter()
            price: float = self.tree(n, trinomial, self.mkt0, prune).price_option(self.opt)
            res["temps"].append(time.perf_counter() - t0)
            res["pas"].append(n)
            res["prix"].append(price)
            res["ecart x pas"].append((price - bs) * n)
        self.conv[name] = res
        print("  ", name, ": prix à", n_list[-1], "pas", price, " Black-Scholes", bs)

    # résultats d'une convergence dans un fichier csv
    def save_csv(self, name: str, file_name: str):
        res: dict = self.conv[name]
        with open(file_name, "w") as f:
            f.write(",".join(res) + "\n")
            for i in range(len(res["pas"])):
                line: str = str(res["pas"][i]) + "," + str(res["prix"][i]) + "," \
                    + str(res["ecart x pas"][i]) + "," + str(res["temps"][i])
                f.write(line + "\n")

    # une courbe par convergence : "key" en fonction du nombre de pas
    def conv_curves(self, names: list, key: str) -> dict:
        curves: dict = {}
        for name in names:
            curves[name] = (self.conv[name]["pas"], self.conv[name][key])
        return curves

    # graphe 1 : (prix - Black-Scholes) x pas, binomial et trinomial, axe des pas en log
    # avec le tunnel de la formule de l'erreur (pointillés : binomial et trinomial)
    def convergence_gap_graph(self) -> Figure:
        fig: Figure = self.figure("(Prix arbre - Black-Scholes) x nombre de pas", "Nombre de pas",
                                  self.conv_curves(["binomial", "trinomial"], "ecart x pas"), True)
        # un point par nombre de pas : on ne relie pas des N éloignés
        for line in fig.axes[0].lines:
            line.set_linestyle("none")
            line.set_marker(".")
        # tunnel donné par la formule de l'erreur du cours
        for trinomial in (False, True):
            g: float = self.gap_x_steps(trinomial)
            color: str = "tab:orange" if trinomial else "tab:blue"
            label: str = "formule " + ("trinomial" if trinomial else "binomial")
            fig.axes[0].axhline(g, color=color, linestyle=":", label=label)
            fig.axes[0].axhline(-g, color=color, linestyle=":")
        fig.axes[0].legend()  # légende refaite : points et pointillés
        return fig

    # graphe 1b : zoom de 1000 à 1100 pas, un point par nombre de pas (comme demandé dans le cours)
    # avec seulement un point tous les 50 pas, le graphe 1 montre une fausse oscillation
    def convergence_zoom_graph(self) -> Figure:
        fig: Figure = self.figure("Zoom : (prix arbre - Black-Scholes) x nombre de pas", "Nombre de pas",
                                  self.conv_curves(["binomial zoom", "trinomial zoom"], "ecart x pas"))
        for line in fig.axes[0].lines:
            line.set_linestyle("none")
            line.set_marker(".")
        fig.axes[0].legend()  # légende refaite : des points, pas des traits
        return fig

    # graphe 7 : prix en fonction du nombre de pas (à partir de 10 pas), avec le prix Black-Scholes
    # sans les tout petits arbres, sinon l'échelle est écrasée par N = 1
    def convergence_price_graph(self) -> Figure:
        curves: dict = {}
        for name in ("binomial", "trinomial"):
            res: dict = self.conv[name]
            steps: list = []
            prices: list = []
            for i in range(len(res["pas"])):
                if res["pas"][i] >= 10:
                    steps.append(res["pas"][i])
                    prices.append(res["prix"][i])
            curves[name] = (steps, prices)
        steps: list = curves["binomial"][0]
        curves["Black-Scholes"] = (steps, [self.bs_price(self.opt)] * len(steps))
        return self.figure("Prix de l'option en fonction du nombre de pas", "Nombre de pas", curves, True)

    # exposant b du temps de calcul : si temps = c x N^b, alors b = log(t2 / t1) / log(N2 / N1)
    # on prend deux points : le premier et le dernier nombre de pas
    def time_exponent(self, steps: list, times: list) -> float:
        return math.log(times[-1] / times[0]) / math.log(steps[-1] / steps[0])

    # vitesse de convergence et temps de calcul (console)
    # 1) si l'écart est en 1/N, |écart| x N ne grandit pas avec N : on compare son maximum
    #    de 10 à n_split pas et au-delà (et à la formule de l'erreur du cours)
    # 2) exposant b du temps de calcul (temps = c x N^b), à partir de n_split pas
    def print_exponents(self, name: str, n_split: int, trinomial: bool):
        res: dict = self.conv[name]
        max_before: float = 0.0  # plus grand |écart| x N de 10 à n_split pas
        max_after: float = 0.0   # plus grand |écart| x N à partir de n_split pas
        steps: list = []         # nombres de pas à partir de n_split
        times: list = []         # temps de calcul correspondants
        for i in range(len(res["pas"])):
            n: int = res["pas"][i]
            gap: float = abs(res["ecart x pas"][i])
            if 10 <= n <= n_split:
                max_before = max(max_before, gap)
            if n >= n_split:
                max_after = max(max_after, gap)
                steps.append(n)
                times.append(res["temps"][i])
        print("  ", name, ": max |écart| x N de 10 à", n_split, "pas", round(max_before, 4),
              " au-delà", round(max_after, 4), " formule", round(self.gap_x_steps(trinomial), 4),
              " temps en N^" + str(round(self.time_exponent(steps, times), 2)))

    # graphe 14 : temps de calcul en fonction du nombre de pas, échelles log, avec l'exposant estimé
    def run_time_graph(self, names: list, n_min: int) -> Figure:
        curves: dict = {}
        for name in names:
            res: dict = self.conv[name]
            steps: list = []
            times: list = []
            for i in range(len(res["pas"])):
                if res["pas"][i] >= n_min:
                    steps.append(res["pas"][i])
                    times.append(res["temps"][i])
            curves[name + " (N^" + str(round(self.time_exponent(steps, times), 2)) + ")"] = (steps, times)
        return self.figure("Temps de calcul en secondes (construction + pricing)", "Nombre de pas",
                           curves, True, True)

    # formule de l'erreur (option ATM forward), écart x nombre de pas :
    # trinomial : 3 S0 exp(rT) sigma² T / (8 racine(2 pi) racine(exp(sigma² T) - 1))
    # binomial : même formule avec 1/2 à la place de 3/8 (l'espace entre les noeuds est plus grand)
    def gap_x_steps(self, trinomial: bool) -> float:
        m: Market = self.mkt0
        t: float = (self.opt.mat_date - self.mdl.price_date).total_seconds() / 86400 / 365
        factor: float = 3 / 8 if trinomial else 1 / 2
        return factor * m.spot_price * math.exp(m.int_rate * t) * m.volatility ** 2 * t \
            / (math.sqrt(2 * math.pi) * math.sqrt(math.exp(m.volatility ** 2 * t) - 1))

    # graphe 5 : temps de calcul du trinomial avec dividende, sans et avec pruning, échelles log
    def pruning_graph(self, n_list: list, prune: float) -> Figure:
        curves: dict = {}
        for p in (0.0, prune):
            times: list = []
            for n in n_list:
                times.append(self.run_time(n, p))
            curves["pruning " + str(p)] = (n_list, times)
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

    # ==================== strike (graphe 2) ====================

    # graphe 2 : écart avec Black-Scholes en fonction du strike, arbre binomial à n pas
    # (pointillés : noeuds de la maturité)
    def strike_gap_graph(self, n: int, k_min: float, k_max: float, k_jump: float) -> Figure:
        tree: Tree = self.tree(n, False, self.mkt0)
        strikes: list = values_between(k_min, k_max, k_jump)
        gaps: list = []
        for k in strikes:
            gaps.append(tree.price_option(self.option(strike=k)) - self.bs_price(self.option(strike=k)))
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

    # ==================== exercice américain (graphe 3) ====================

    # graphe 3 : américaine - européenne en fonction du taux, call et put, sans et avec dividende
    def rate_graph(self, n: int, r_min: float, r_max: float, r_jump: float) -> Figure:
        rates: list = values_between(r_min, r_max, r_jump)
        curves: dict = {}
        for div in (0.0, self.mkt.div_amount):
            for call_put in (CALL, PUT):
                diffs: list = []
                rounded: list = []  # pour l'affichage dans la console
                for r in rates:
                    d: float = self.amer_minus_euro(n, self.market(r, div), call_put)
                    diffs.append(d)
                    rounded.append(round(d, 4))
                curves[call_put + ", dividende " + str(div)] = (rates, diffs)
                print("  ", call_put, "dividende", div, ":", rounded)
        fig: Figure = self.figure("Américaine - européenne en fonction du taux", "Taux d'intérêt", curves)
        fig.axes[0].axvline(0.0, color="black", linewidth=0.5)
        return fig

    # prix américain - prix européen sur le même arbre trinomial
    def amer_minus_euro(self, n: int, mkt: Market, call_put: str) -> float:
        tree: Tree = self.tree(n, True, mkt)
        return tree.price_option(self.option(call_put, AMER)) - tree.price_option(self.option(call_put, EURO))

    # ==================== dividende ====================

    # effet du dividende avec un taux = 0 : call et put européens sans et avec dividende (console)
    def print_dividend_effect(self, n: int):
        for div in (0.0, self.mkt.div_amount):
            tree: Tree = self.tree(n, True, self.market(0.0, div))
            call: float = tree.price_option(self.option(CALL, EURO))
            put: float = tree.price_option(self.option(PUT, EURO))
            print("   dividende", div, ": call", round(call, 6), " put", round(put, 6),
                  " call - put", round(call - put, 6))

    # ==================== grecques (graphe 12) ====================

    # graphe 12 : delta de l'arbre trinomial en fonction du nombre de pas, avec les deux méthodes,
    # comparé à Black-Scholes
    def delta_graph(self, n_list: list) -> Figure:
        nodes: list = []
        shift: list = []
        for n in n_list:
            g: Greeks = Greeks(self.mkt0, Model(self.mdl.price_date, n), self.opt)
            nodes.append(g.delta())
            shift.append(g.delta_shift())
        bs: float = BlackScholes(self.mkt0, self.mdl.price_date, self.opt).delta()
        return self.figure("Delta en fonction du nombre de pas", "Nombre de pas",
                           {"arbre, noeuds autour de la racine": (n_list, nodes),
                            "arbre, choc de 1 % sur le spot": (n_list, shift),
                            "Black-Scholes": (n_list, [bs] * len(n_list))})

    # grecques de l'arbre trinomial à n pas (console), pour l'option de l'étude et une option OTM :
    # sans dividende comparées à Black-Scholes, puis avec le dividende du marché
    # (avec un dividende discret, Black-Scholes ne donne pas de référence)
    # vega, vanna et vomma pour 1 % de vol, theta pour 1 jour
    def print_greeks(self, n: int, otm_strike: float):
        for opt in (self.opt, self.option(strike=otm_strike)):
            print("  ", opt)
            self.print_tree_greeks("arbre sans dividende", self.mkt0, n, opt)
            self.print_bs_greeks(opt)
            self.print_tree_greeks("arbre avec dividende", self.mkt, n, opt)

    # une ligne de grecques de l'arbre trinomial à n pas sur le marché mkt
    def print_tree_greeks(self, label: str, mkt: Market, n: int, opt: OptionTrade):
        g: Greeks = Greeks(mkt, Model(self.mdl.price_date, n), opt)
        Study.print_greeks_line(label + " " + str(n) + " pas", g.delta(), g.gamma(), g.vega(), g.vomma(),
                                g.vanna(), g.theta())
        print("      ", label, "choc de 1 % sur le spot : delta", round(g.delta_shift(), 6),
              " gamma", round(g.gamma_shift(), 6))

    # une ligne de grecques Black-Scholes, marché sans dividende (theta : prix demain - prix aujourd'hui)
    def print_bs_greeks(self, opt: OptionTrade):
        bs: BlackScholes = BlackScholes(self.mkt0, self.mdl.price_date, opt)
        tomorrow: datetime.datetime = self.mdl.price_date + datetime.timedelta(days=1)
        theta: float = BlackScholes(self.mkt0, tomorrow, opt).price() - bs.price()
        Study.print_greeks_line("Black-Scholes sans dividende", bs.delta(), bs.gamma(), bs.vega(), bs.vomma(),
                                bs.vanna(), theta)

    # affiche les six grecques ; vega, vomma et vanna sont données pour 1 unité de vol
    @staticmethod
    def print_greeks_line(label: str, delta: float, gamma: float, vega: float, vomma: float,
                          vanna: float, theta: float):
        print("     ", label, ": delta", round(delta, 6), " gamma", round(gamma, 6),
              " vega", round(vega * 0.01, 6), " vomma", round(vomma * 0.0001, 6),
              " vanna", round(vanna * 0.01, 6), " theta", round(theta, 6))

    # ==================== frontière d'exercice (graphe 13) ====================

    # graphe 13 : frontière d'exercice de l'option opt, sans et avec dividende, arbre trinomial à n pas
    # put : on exerce sous la courbe ; call : on exerce au-dessus
    # une courbe seulement s'il y a un exercice avant la maturité, None s'il n'y en a dans aucun des deux marchés
    def exercise_boundary_graph(self, n: int, prune: float, opt: OptionTrade) -> Figure:
        curves: dict = {}
        for mkt in (self.mkt0, self.mkt):
            name: str = opt.opt_type + " " + opt.exercice + ", dividende " + str(mkt.div_amount)
            dates, s_star, early = self.boundary(self.tree(n, True, mkt, prune), opt)
            if early:
                curves[name] = (dates, s_star)
            first: str = "le " + str(dates[0].date()) if early else "aucun"
            print("  ", name, ": premier exercice anticipé :", first)
        if len(curves) == 0:
            return None
        fig: Figure = self.figure("Frontière d'exercice du " + opt.opt_type + " " + opt.exercice
                                  + " (" + str(n) + " pas)", "Date", curves)
        for line in fig.axes[0].lines:
            line.set_linestyle("none")  # un point par date, sans trait : on ne relie pas des dates éloignées
            line.set_marker(".")
        fig.axes[0].axvline(self.mkt.div_ex_date, color="grey", linestyle="--", label="date ex-dividende")
        fig.axes[0].axhline(opt.strike, color="black", linestyle=":", label="strike")
        fig.axes[0].legend(loc="upper left")  # en haut à gauche : la légende ne cache pas la courbe
        return fig

    # prix critique S* à chaque date où l'option est exercée, et True s'il y a un exercice avant la maturité
    def boundary(self, tree: TrinomialTree, opt: OptionTrade) -> tuple:
        tree.price_option(opt)
        dates: list = []
        s_star: list = []
        early: bool = False
        trunk: TrinomialTrunkNode = tree.root
        while trunk is not None:
            price: float = self.limit_exercised(trunk, opt)
            if price is not None:
                dates.append(trunk.column_date)
                s_star.append(price)
                if trunk.next_mid is not None:  # pas la dernière date : exercice anticipé
                    early = True
            trunk = trunk.next_mid
        return dates, s_star, early

    # prix du noeud exercé le plus proche de la zone où l'on garde l'option, None si aucun noeud n'est exercé
    def limit_exercised(self, trunk: TrinomialTrunkNode, opt: OptionTrade) -> float:
        n: Node = trunk
        if opt.is_a_put():
            # put : on exerce quand S est bas, on part du haut de la colonne et on descend
            while n.unode is not None:
                n = n.unode
            while n is not None and not n.is_exercised():
                n = n.dnode
        else:
            # call : on exerce quand S est haut, on part du bas de la colonne et on monte
            while n.dnode is not None:
                n = n.dnode
            while n is not None and not n.is_exercised():
                n = n.unode
        if n is None:
            return None
        return n.under_price

    # ==================== compléments : vus en cours, hors énoncé (graphes 6, 11 et 8) ====================

    # graphe 6 : dernier pas, (prix - Black-Scholes) x pas pour les quatre méthodes
    def last_step_graph(self, n_list: list) -> Figure:
        bs: float = self.bs_price(self.opt)
        curves: dict = {}
        for name in Study.LAST_STEP_METHODS:
            trinomial, last_step_bs = Study.LAST_STEP_METHODS[name]
            gaps: list = []
            for n in n_list:
                price: float = self.tree(n, trinomial, self.mkt0).price_option(self.opt, last_step_bs)
                gaps.append((price - bs) * n)
            curves[name] = (n_list, gaps)
            print("  ", name, ": écart x pas à", n_list[-1], "pas =", round(gaps[-1], 6))
        return self.figure("Dernier pas : (prix arbre - Black-Scholes) x nombre de pas", "Nombre de pas",
                           curves, True)

    # graphe 11 : dernier pas, écart avec Black-Scholes en fonction du strike, arbres à n pas
    def last_step_strike_graph(self, n: int, k_min: float, k_max: float, k_jump: float) -> Figure:
        strikes: list = values_between(k_min, k_max, k_jump)
        curves: dict = {}
        for name in Study.LAST_STEP_METHODS:
            trinomial, last_step_bs = Study.LAST_STEP_METHODS[name]
            tree: Tree = self.tree(n, trinomial, self.mkt0)
            gaps: list = []
            for k in strikes:
                price: float = tree.price_option(self.option(strike=k), last_step_bs)
                gaps.append(price - self.bs_price(self.option(strike=k)))
            curves[name] = (strikes, gaps)
        return self.figure("Dernier pas : prix arbre - Black-Scholes (" + str(n) + " pas)", "Strike", curves)

    # graphe 8 : proba d'être dans la zone d'exercice, par date, options américaines, arbre trinomial à n pas
    def exercise_graph(self, n: int) -> Figure:
        curves: dict = {}
        for call_put, mkt in ((PUT, self.mkt0), (PUT, self.mkt), (CALL, self.mkt)):
            name: str = call_put + " américain, dividende " + str(mkt.div_amount)
            curves[name] = self.exercise_probas(self.tree(n, True, mkt), self.option(call_put, AMER))
        return self.figure("Probabilité d'être dans la zone d'exercice, par date (" + str(n) + " pas)",
                           "Date", curves)

    # dates du tronc et proba d'être dans la zone d'exercice à chaque date, après le pricing de opt
    # (somme des probas des noeuds exercés, comme dans le cours : ce n'est pas la proba d'exercer
    # pour la première fois à cette date, car un chemin a pu être exercé avant)
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
