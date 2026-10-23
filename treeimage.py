from __future__ import annotations
import math
from vectorimage import VectorImage, to_hexa_two_char
from tree import Tree
from node import Node
from optiontrade import OptionTrade


# image svg de l'arbre binomial ou trinomial : un cercle par noeud, coloré selon la valeur de l'option
class TreeImage(VectorImage):

    def __init__(self, width: int, height: int, tree: Tree, opt: OptionTrade):
        super().__init__(width, height)
        self.tree: Tree = tree
        self.opt: OptionTrade = opt
        # espace horizontal entre deux dates, échelle verticale pour ln(S / S0)
        self.margin: int = 40
        self.dx: float = (width - 2 * self.margin) / tree.nb_steps
        self.dy: float = (height / 2 - self.margin) / self._max_ln()
        self.node_radius: int = max(min(int(self.dx / 6), 12), 1)
        self._build_image()

    # plus grand ln(S / S0) à la dernière date, entre le noeud du haut et celui du bas
    def _max_ln(self) -> float:
        top: Node = self.tree.last_trunk
        while top.unode is not None:
            top = top.unode
        bottom: Node = self.tree.last_trunk
        while bottom.dnode is not None:
            bottom = bottom.dnode
        s0: float = self.tree.root.under_price
        return max(math.log(top.under_price / s0), math.log(s0 / bottom.under_price))

    # tous les noeuds, date par date, en partant du noeud du haut de chaque colonne
    def _build_image(self):
        top: Node = self.tree.root
        for i in range(self.tree.nb_steps + 1):
            n: Node = top
            while n is not None:
                self._build_image_node(i, n)
                n = n.dnode
            if i < self.tree.nb_steps:
                top = self.tree.next_column_top(top)

    # cercle du noeud (bord rouge si l'option est exercée), segments vers les noeuds suivants
    def _build_image_node(self, i: int, n: Node):
        x, y = self._node_position(i, n)
        stroke_color: str = 'black'
        if self._is_exercised(n):
            stroke_color = 'red'
        self.add_end(self.circle(x, y, self.node_radius, stroke_color, self._node_color(n), 1))
        self._build_image_segment(i, x, y, n.next_up)
        self._build_image_segment(i, x, y, n.next_mid)
        self._build_image_segment(i, x, y, n.next_down)
        if self._labels_fit():
            self._add_labels(x, y, n)

    # segment du noeud (x, y) vers un de ses noeuds suivants, s'il existe
    def _build_image_segment(self, i: int, x: int, y: int, next_n: Node):
        if next_n is not None:
            next_x, next_y = self._node_position(i + 1, next_n)
            self.add_beginning(self.segment(x, y, next_x, next_y, '#C0C0C0', 1))

    # x : une colonne par date, y : ln(S / S0) avec la racine au milieu
    def _node_position(self, i: int, n: Node) -> tuple:
        x: int = int(self.margin + i * self.dx)
        y: int = int(self.height / 2 - math.log(n.under_price / self.tree.root.under_price) * self.dy)
        return x, y

    # blanc si l'option ne vaut rien, puis du bleu (petite valeur) au vert (valeur proche du spot)
    def _node_color(self, n: Node) -> str:
        if n.option_price < 0.01:
            return '#FFFFFF'
        ln_opt_value: float = math.log(n.option_price)
        blue: float = math.log(0.01)
        green: float = math.log(self.tree.root.under_price)
        g = int(max(min(255 * (ln_opt_value - blue) / (green - blue), 255), 0))
        b = int(max(min(255 * (green - ln_opt_value) / (green - blue), 255), 0))
        return '#00' + to_hexa_two_char(g) + to_hexa_two_char(b)

    # option américaine exercée avant la maturité : la valeur est le payoff
    def _is_exercised(self, n: Node) -> bool:
        payoff: float = self.opt.pay_off(n.under_price)
        return self.opt.is_american() and n.next_up is not None and payoff > 0 and n.option_price == payoff

    # les deux nombres d'un noeud ont besoin d'environ 35 pixels entre deux voisins
    def _labels_fit(self) -> bool:
        x, y_up = self._node_position(1, self.tree.root.next_up)
        x, y_down = self._node_position(1, self.tree.root.next_up.dnode)
        return y_down - y_up >= 35

    # prix du sous-jacent au-dessus du noeud, valeur de l'option en dessous
    def _add_labels(self, x: int, y: int, n: Node):
        self.add_end(self.text(x, y - self.node_radius - 4, str(round(n.under_price, 2)), 'black'))
        self.add_end(self.text(x, y + self.node_radius + 13, str(round(n.option_price, 2)), 'grey'))
