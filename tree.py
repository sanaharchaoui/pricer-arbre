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
        self.mkt: Market = mkt
        self.nb_steps: int = mdl.steps
        self.delta_t: float = (mat - mdl.price_date).days / 365 / mdl.steps

        # paramètres de l'arbre binomial
        a: float = (1 + math.exp(mkt.volatility ** 2 * self.delta_t)) / 2
        self.sqrt_alpha: float = a + math.sqrt(a ** 2 - 1)
        self.p_up: float = 1 / (1 + self.sqrt_alpha)
        self.p_down: float = 1 - self.p_up
        self.df: float = math.exp(-mkt.int_rate * self.delta_t)
        self.root: TrunkNode = TrunkNode(mkt.spot_price, self, None)
        self.last_trunk: TrunkNode = None

    # on descend la colonne depuis le noeud du haut : chaque noeud prend comme next_up
    # le next_down du noeud au-dessus, et crée son propre next_down
    def build_next_column(self, top: Node, trunk_parent: Node, trunk: TrunkNode):
        top.next_up = Node(top.forward() * self.sqrt_alpha, self)
        n: Node = top
        while n is not None:
            if n.unode is not None:
                n.next_up = n.unode.next_down
            if n is trunk_parent:
                n.next_down = TrunkNode(n.forward() / self.sqrt_alpha, self, trunk)
            else:
                n.next_down = Node(n.forward() / self.sqrt_alpha, self)
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
                n.calc_bs_price(opt)
            else:
                n.calc_option_price(opt)
            n = n.dnode

    # backward : de la dernière date jusqu'à la racine, en suivant le tronc
    # last_step_bs = False : on commence à la date N avec la valeur intrinsèque
    # last_step_bs = True : on commence à la date N-1 avec Black-Scholes sur le dernier pas
    def price_option(self, opt: OptionTrade, last_step_bs: bool = False) -> float:
        trunk: TrunkNode = self.last_trunk
        if last_step_bs:
            trunk = trunk.prec_trunk
            self.price_column(trunk, opt, True)
            trunk = trunk.prec_trunk
        while trunk is not None:
            self.price_column(trunk, opt)
            trunk = trunk.prec_trunk
        return self.root.option_price

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

    # prix du sous-jacent et valeur de l'option de chaque noeud, date par date
    def print_tree(self):
        top: Node = self.root
        for i in range(self.nb_steps + 1):
            print("date", i)
            n: Node = top
            while n is not None:
                print("   S =", round(n.under_price, 4), " option =", round(n.option_price, 4))
                n = n.dnode
            if i < self.nb_steps:
                top = self.next_column_top(top)
