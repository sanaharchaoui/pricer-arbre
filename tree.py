from __future__ import annotations
import datetime as datetime
import math
from market import Market
from model import Model
from node import Node
from trunknode import TrunkNode
from optiontrade import OptionTrade


class Tree(object):

    def __init__(self, mkt: Market, mdl: Model, mat: datetime.datetime):
        if not (mkt.definition_ok and mat > mdl.price_date and mdl.steps >= 1):
            raise ValueError("paramètres incorrects : prix ou vol nul ou négatif, dividende négatif,"
                             " maturité avant la date de pricing, ou moins d'un pas")
        if mkt.has_dividend(mdl.price_date, mat) and not self.handles_dividend():
            raise ValueError("l'arbre binomial ne prend pas en compte le dividende : utiliser l'arbre trinomial")
        self.mkt: Market = mkt
        self.price_date: datetime.datetime = mdl.price_date
        self.mat_date: datetime.datetime = mat
        self.nb_steps: int = mdl.steps + self.steps_before_pricing()
        self.delta_t: float = (mat - mdl.price_date).total_seconds() / 86400 / 365 / mdl.steps

        # paramètres de l'arbre binomial (cours p. 8) : alpha_bin est le alpha du cours,
        # noeud du haut = forward x alpha_bin, noeud du bas = forward / alpha_bin
        a: float = (1 + math.exp(mkt.volatility ** 2 * self.delta_t)) / 2
        self.alpha_bin: float = a + math.sqrt(a ** 2 - 1)
        self.p_up: float = 1 / (1 + self.alpha_bin)
        self.p_down: float = 1 - self.p_up
        self.df: float = math.exp(-mkt.int_rate * self.delta_t)
        self.root: TrunkNode = self.new_root()
        self.last_trunk: TrunkNode = None

    # nombre de pas ajoutés avant la date de pricing : aucun pour l'arbre binomial
    def steps_before_pricing(self) -> int:
        return 0

    # l'arbre binomial ne sait pas gérer un dividende discret (le trinomial, oui)
    def handles_dividend(self) -> bool:
        return False

    # racine de l'arbre : le spot à la date de pricing
    def new_root(self) -> TrunkNode:
        return TrunkNode(self.mkt.spot_price, self, None)

    # noeud de la date de pricing, dont la valeur est le prix de l'option
    def price_node(self) -> Node:
        return self.root

    # on descend la colonne depuis le noeud du haut : chaque noeud prend comme next_up
    # le next_down du noeud au-dessus, et crée son propre next_down
    def build_next_column(self, top: Node, trunk_parent: Node, trunk: TrunkNode):
        top.next_up = Node(top.forward() * self.alpha_bin, self)
        n: Node = top
        while n is not None:
            if n.unode is not None:
                n.next_up = n.unode.next_down
            if n is trunk_parent:
                n.next_down = TrunkNode(n.forward() / self.alpha_bin, self, trunk)
            else:
                n.next_down = Node(n.forward() / self.alpha_bin, self)
            n.next_up.dnode = n.next_down
            n.next_down.unode = n.next_up
            n = n.dnode

    # le tronc descend, puis monte, puis descend et ainsi de suite
    # le nouveau noeud du tronc est toujours le next_down de trunk_parent
    def build_tree(self):
        trunk: TrunkNode = self.root
        top: Node = self.root
        for i in range(self.nb_steps):
            trunk_parent: Node = trunk
            if i % 2 == 1:
                trunk_parent = trunk.unode
            self.build_next_column(top, trunk_parent, trunk)
            trunk = trunk_parent.next_down
            top = top.next_up
        self.last_trunk = trunk

    # on monte en haut de la colonne, puis on price chaque noeud en descendant
    # (avec Black-Scholes si c'est la colonne N-1 de la deuxième méthode)
    def price_column(self, trunk: TrunkNode, opt: OptionTrade, with_bs: bool = False):
        n: Node = trunk
        while n.unode is not None:
            n = n.unode
        while n is not None:
            if with_bs:
                n.calc_bs_price(opt, trunk)
            else:
                n.calc_option_price(opt)
            n = n.dnode

    # backward : de la dernière date jusqu'à la racine, en suivant le tronc
    # la date N vaut toujours la valeur intrinsèque, puis :
    # last_step_bs = False : la date N-1 est la moyenne actualisée des noeuds suivants
    # last_step_bs = True : la date N-1 est le prix Black-Scholes sur le dernier pas
    def price_option(self, opt: OptionTrade, last_step_bs: bool = False) -> float:
        if not opt.definition_ok:
            raise ValueError("option incorrecte : type, exercice ou strike")
        self.price_column(self.last_trunk, opt)
        trunk: TrunkNode = self.last_trunk.prec_trunk
        if last_step_bs and trunk is not None:
            self.price_column(trunk, opt, True)
            trunk = trunk.prec_trunk
        while trunk is not None:
            self.price_column(trunk, opt)
            trunk = trunk.prec_trunk
        return self.price_node().option_price

    # noeud du haut de la colonne suivante : next_up du noeud du haut, ou son next_mid s'il est pruné
    def next_column_top(self, top: Node) -> Node:
        n: Node = top.next_up
        if n is None:
            n = top.next_mid
        while n.unode is not None:
            n = n.unode
        return n

    # nombre de noeuds, colonne par colonne, du dernier tronc jusqu'à la racine
    def count_nodes(self) -> int:
        count: int = 0
        trunk: TrunkNode = self.last_trunk
        while trunk is not None:
            n: Node = trunk
            while n.unode is not None:
                n = n.unode
            while n is not None:
                count = count + 1
                n = n.dnode
            trunk = trunk.prec_trunk
        return count
