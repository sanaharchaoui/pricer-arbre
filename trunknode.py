from __future__ import annotations
from node import Node


class TrunkNode(Node):

    def __init__(self, price: float, tree: Tree, prec: TrunkNode):
        Node.__init__(self, price, tree)
        # tous les attributs de TrunkNode sont hérités de node
        # on ajoute un seul attribut qui est le noeud du tronc qui précède ce noeud
        self.prec_trunk: TrunkNode = prec
