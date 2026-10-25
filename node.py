from __future__ import annotations
import math
from market import Market
from optiontrade import OptionTrade
from blackscholes import BlackScholes


class Node(object):

    def __init__(self, price: float, tree: Tree):
        self.under_price: float = price
        self.tree: Tree = tree

        # noeuds à la date suivante
        self.next_up: Node = None
        self.next_mid: Node = None # que pour l'arbre trinomial
        self.next_down: Node = None

        # voisins à la même date
        self.unode: Node = None # noeud du dessus
        self.dnode: Node = None # noeud du dessous

        # valeur de l'option sur ce noeud
        self.option_price: float = None

    # prix forward sur un pas : S x exp(r delta_t)
    def forward(self) -> float:
        return self.under_price * math.exp(self.tree.mkt.int_rate * self.tree.delta_t)

    # dernière date : payoff, sinon on calcule la moyenne actualisée des deux noeuds suivants
    def calc_option_price(self, opt: OptionTrade):
        if self.next_up is None:
            self.option_price = opt.pay_off(self.under_price)
        else:
            t: Tree = self.tree
            value: float = t.df * (t.p_up * self.next_up.option_price
                                   + t.p_down * self.next_down.option_price)
            if opt.is_american():
                value = max(value, opt.pay_off(self.under_price))
            self.option_price = value

    # AUTRE MÉTHODE POUR CALCULER LE PRIX DE L'OPTION AU DERNIER NOEUD
    # date N-1 : il reste un seul pas avant la maturité, on prend le prix Black-Scholes sur delta_t
    # (spot = prix du noeud moins le dividende du dernier pas s'il y en a un)
    # pour une américaine on garde le max avec la valeur intrinsèque
    def calc_bs_price(self, opt: OptionTrade, trunk: TrunkNode):
        spot: float = self.under_price - self.dividend_pv(trunk)
        mkt: Market = Market(spot, self.tree.mkt.volatility, self.tree.mkt.int_rate, 0.0, None)
        bs: BlackScholes = BlackScholes(mkt, opt.mat_date, opt)
        bs.mat_time = self.tree.delta_t
        value: float = bs.price()
        if opt.is_american():
            value = max(value, opt.pay_off(self.under_price))
        self.option_price = value

    # valeur actuelle du dividende payé pendant le pas suivant : 0 dans l'arbre binomial (pas de dividende)
    def dividend_pv(self, trunk: TrunkNode) -> float:
        return 0.0
