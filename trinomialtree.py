from __future__ import annotations
import datetime as datetime
import math
import warnings
from market import Market
from model import Model
from optiontrade import OptionTrade
from tree import Tree
from trinomialnode import TrinomialNode
from trinomialtrunknode import TrinomialTrunkNode


# arbre trinomial : même pricing backward que le Tree binomial (price_option), seule la construction change
class TrinomialTree(Tree):

    # binomial_alpha : espace entre les noeuds comme le binomial
    # around_root : l'arbre commence un pas avant la date de pricing, pour avoir trois noeuds
    # (up, milieu, down) à la date de pricing : ils servent au delta et au gamma
    def __init__(self, mkt: Market, mdl: Model, mat: datetime.datetime, binomial_alpha: bool = False,
                 around_root: bool = False):
        self.around_root: bool = around_root
        # noeuds dont une proba de transition est négative (comptés pendant la construction) :
        # leur nombre et la somme de leurs probas d'atteinte
        self.negative_nodes: int = 0
        self.negative_weight: float = 0.0
        # seuil de pruning sur la proba d'atteindre un noeud, 0 = pas de pruning
        self.prune_proba: float = mdl.prune_proba
        Tree.__init__(self, mkt, mdl, mat)
        self.alpha: float = self.space_between_nodes(binomial_alpha)

    # l'arbre trinomial gère le dividende discret
    def handles_dividend(self) -> bool:
        return True

    # around_root : un pas de plus, avant la date de pricing
    def steps_before_pricing(self) -> int:
        return 1 if self.around_root else 0

    # racine : le spot à la date de pricing, ou, si around_root, le spot actualisé un pas avant
    # (son noeud du milieu à la date de pricing vaut alors le spot)
    def new_root(self) -> TrinomialTrunkNode:
        if self.around_root:
            start: datetime.datetime = self.price_date - datetime.timedelta(days=self.delta_t * 365)
            root: TrinomialTrunkNode = TrinomialTrunkNode(self.mkt.spot_price * self.df, self, None, start)
        else:
            root = TrinomialTrunkNode(self.mkt.spot_price, self, None, self.price_date)
        root.proba = 1.0
        return root

    # noeud de la date de pricing : la racine, ou le noeud du milieu de la racine si around_root
    def price_node(self) -> TrinomialNode:
        if self.around_root:
            return self.root.next_mid
        return self.root

    # espace entre les noeuds : sqrt(3) écarts-types (Hull & White), ou celui du binomial au carré
    def space_between_nodes(self, binomial_alpha: bool) -> float:
        if binomial_alpha:
            return self.sqrt_alpha ** 2
        return math.exp(self.mkt.volatility * math.sqrt(3 * self.delta_t))

    # le tronc passe par les noeuds du milieu, date par date
    # contrainte du modèle : probas positives, on prévient si des noeuds probables ne la respectent pas
    def build_tree(self):
        trunk: TrinomialTrunkNode = self.root
        for i in range(self.nb_steps):
            trunk = self.build_next_column(trunk)
        self.last_trunk = trunk
        if self.negative_weight > 1e-6:
            warnings.warn("probas de transition négatives sur " + str(self.negative_nodes) + " noeuds atteints"
                          " avec une proba totale de " + str(round(self.negative_weight, 6))
                          + " (dividende grand devant le prix de l'action) : le prix est moins fiable")

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
            if min(n.p_up, n.p_mid, n.p_down) < 0:
                self.negative_nodes = self.negative_nodes + 1
                self.negative_weight = self.negative_weight + n.proba
        n.spread_proba()

    # pruning : un noeud très improbable va seulement vers son noeud du milieu, avec une proba = 1
    # le noeud du milieu est le plus proche de son espérance, créé s'il n'existe pas encore
    # (sinon, en haut de l'arbre, il serait envoyé trop bas et la queue de la loi se tasserait)
    def link_pruned_node(self, n: TrinomialNode, guess: TrinomialNode, e: float):
        n.next_mid = self.closest_node(guess, e)
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
        return self.negative_weight

    # proba d'exercice sur la colonne de ce tronc, après le pricing de opt :
    # somme des probas des noeuds où l'option est exercée
    def column_exercise_proba(self, trunk: TrinomialTrunkNode, opt: OptionTrade) -> float:
        proba: float = 0.0
        n: TrinomialNode = trunk
        while n.unode is not None:
            n = n.unode
        while n is not None:
            if n.is_exercised():
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
