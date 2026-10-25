from __future__ import annotations
import datetime as datetime
import math
from market import Market
from model import Model
from optiontrade import OptionTrade
from tree import Tree
from trinomialnode import TrinomialNode
from trinomialtrunknode import TrinomialTrunkNode


# arbre trinomial : même pricing backward que le Tree binomial (price_option), seule la construction change
class TrinomialTree(Tree):

    def __init__(self, mkt: Market, mdl: Model, mat: datetime.datetime, binomial_alpha: bool = False):
        self.mkt: Market = mkt
        self.nb_steps: int = mdl.steps
        self.delta_t: float = (mat - mdl.price_date).days / 365 / mdl.steps
        self.df: float = math.exp(-mkt.int_rate * self.delta_t)
        self.alpha: float = self.space_between_nodes(binomial_alpha)
        # seuil de pruning sur la proba d'atteindre un noeud, 0 = pas de pruning
        self.prune_proba: float = mdl.prune_proba
        self.root: TrinomialTrunkNode = TrinomialTrunkNode(mkt.spot_price, self, None, mdl.price_date)
        self.root.proba = 1.0
        self.last_trunk: TrinomialTrunkNode = None

    # espace entre les noeuds : sqrt(3) écarts-types (Hull & White), ou 2 comme le binomial
    def space_between_nodes(self, binomial_alpha: bool) -> float:
        if binomial_alpha:
            a: float = (1 + math.exp(self.mkt.volatility ** 2 * self.delta_t)) / 2
            return (a + math.sqrt(a ** 2 - 1)) ** 2
        return math.exp(self.mkt.volatility * math.sqrt(3 * self.delta_t))

    # le tronc passe par les noeuds du milieu, date par date
    def build_tree(self):
        trunk: TrinomialTrunkNode = self.root
        for i in range(self.nb_steps):
            trunk = self.build_next_column(trunk)
        self.last_trunk = trunk

    # le nouveau tronc est l'espérance de l'ancien tronc, puis on relie les noeuds au-dessus et en dessous :
    # chacun commence sa recherche à partir du noeud du milieu de son voisin, déjà relié
    def build_next_column(self, trunk: TrinomialTrunkNode) -> TrinomialTrunkNode:
        new_trunk: TrinomialTrunkNode = TrinomialTrunkNode(trunk.expectation(trunk), self, trunk,
                                                           trunk.next_date())
        self.link_node(trunk, new_trunk, trunk)
        n: TrinomialNode = trunk.unode
        while n is not None:
            self.link_node(n, n.dnode.next_mid, trunk)
            n = n.unode
        n = trunk.dnode
        while n is not None:
            self.link_node(n, n.unode.next_mid, trunk)
            n = n.dnode
        return new_trunk

    # le noeud du milieu de n est le noeud suivant le plus proche de son espérance
    def link_node(self, n: TrinomialNode, guess: TrinomialNode, trunk: TrinomialTrunkNode):
        e: float = n.expectation(trunk)
        if n.proba < self.prune_proba:
            self.link_pruned_node(n, guess, e)
        else:
            n.next_mid = self.closest_node(guess, e)
            n.next_up = self.node_above(n.next_mid)
            n.next_down = self.node_below(n.next_mid)
            n.calc_probabilities(e, n.variance(trunk))
        n.spread_proba()

    # pruning : un noeud très improbable va seulement vers son noeud du milieu,
    # avec une proba = 1, choisi parmi les noeuds qui existent : on ne crée pas de nouveau noeud
    def link_pruned_node(self, n: TrinomialNode, guess: TrinomialNode, e: float):
        m: TrinomialNode = guess
        while m.unode is not None and e > m.under_price * (1 + self.alpha) / 2:
            m = m.unode
        while m.dnode is not None and e < m.under_price * (1 + 1 / self.alpha) / 2:
            m = m.dnode
        n.next_mid = m
        n.p_mid = 1.0

    # on monte ou on descend depuis guess tant qu'un voisin est plus proche de e
    # (1 + alpha) / 2 est le point au milieu entre un noeud et son voisin du dessus
    def closest_node(self, m: TrinomialNode, e: float) -> TrinomialNode:
        while e > m.under_price * (1 + self.alpha) / 2:
            m = self.node_above(m)
        while e < m.under_price * (1 + 1 / self.alpha) / 2:
            m = self.node_below(m)
        return m

    # monitoring de la colonne de ce tronc : somme des probas d'atteindre ses noeuds (doit faire 1),
    # prix moyen du sous-jacent (doit être le prix du tronc) et valeur moyenne de l'option
    def column_monitoring(self, trunk: TrinomialTrunkNode) -> tuple:
        proba: float = 0.0
        under: float = 0.0
        option: float = 0.0
        n: TrinomialNode = trunk
        while n.unode is not None:
            n = n.unode
        while n is not None:
            proba = proba + n.proba
            under = under + n.proba * n.under_price
            option = option + n.proba * n.option_price
            n = n.dnode
        return proba, under, option

    # poids des noeuds qui ont une proba de transition négative : somme de leurs probas d'atteinte
    # (arrive quand le dividende est grand devant le prix de l'action sur une partie de l'arbre)
    def negative_proba_weight(self) -> float:
        weight: float = 0.0
        trunk: TrinomialTrunkNode = self.root
        while trunk.next_mid is not None:
            n: TrinomialNode = trunk
            while n.unode is not None:
                n = n.unode
            while n is not None:
                if n.next_up is not None and min(n.p_up, n.p_mid, n.p_down) < 0:
                    weight = weight + n.proba
                n = n.dnode
            trunk = trunk.next_mid
        return weight

    # proba d'exercice sur la colonne de ce tronc, après le pricing de opt :
    # somme des probas des noeuds où l'option est exercée
    def column_exercise_proba(self, trunk: TrinomialTrunkNode, opt: OptionTrade) -> float:
        proba: float = 0.0
        n: TrinomialNode = trunk
        while n.unode is not None:
            n = n.unode
        while n is not None:
            if n.is_exercised(opt):
                proba = proba + n.proba
            n = n.dnode
        return proba

    # monitoring toutes les "every" dates, après le pricing : pour une option européenne,
    # la valeur moyenne actualisée de l'option doit être le prix de la racine à chaque date
    def print_monitoring(self, every: int):
        trunk: TrinomialTrunkNode = self.root
        for i in range(self.nb_steps + 1):
            proba, under, option = self.column_monitoring(trunk)
            if i % every == 0 or i == self.nb_steps:
                print("date", i, " : somme des probas", round(proba, 12), " S moyen", round(under, 6),
                      " tronc", round(trunk.under_price, 6),
                      " option moyenne actualisée", round(option * self.df ** i, 6))
            trunk = trunk.next_mid

    # voisin au-dessus de m, créé s'il n'existe pas encore
    def node_above(self, m: TrinomialNode) -> TrinomialNode:
        if m.unode is None:
            m.unode = TrinomialNode(m.under_price * self.alpha, self)
            m.unode.dnode = m
        return m.unode

    # voisin en dessous de m, créé s'il n'existe pas encore
    def node_below(self, m: TrinomialNode) -> TrinomialNode:
        if m.dnode is None:
            m.dnode = TrinomialNode(m.under_price / self.alpha, self)
            m.dnode.unode = m
        return m.dnode
