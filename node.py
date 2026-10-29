from __future__ import annotations
import datetime as datetime
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

        # valeur de l'option sur ce noeud, et True si l'option y est exercée
        self.option_price: float = None
        self.exercised: bool = False

    # prix forward sur un pas : S x exp(r delta_t)
    def forward(self) -> float:
        return self.under_price * math.exp(self.tree.mkt.int_rate * self.tree.delta_t)

    # valeur si on garde l'option : moyenne actualisée des deux noeuds suivants
    def hold_value(self) -> float:
        t: Tree = self.tree
        return t.df * (t.p_up * self.next_up.option_price + t.p_down * self.next_down.option_price)

    # dernière date : payoff, sinon valeur si on garde (ou le payoff si on exerce une américaine)
    def calc_option_price(self, opt: OptionTrade):
        if self.next_up is None:
            self.set_value(opt, opt.pay_off(self.under_price), True)
        else:
            self.set_value(opt, self.hold_value(), opt.is_american())

    # valeur du noeud à partir de la valeur si on garde (hold) :
    # à maturité on exerce si le payoff est positif ; avant, seulement si on peut exercer (can_exercise)
    # et si exercer vaut strictement plus que garder (pas d'exercice en cas d'égalité)
    def set_value(self, opt: OptionTrade, hold: float, can_exercise: bool):
        payoff: float = opt.pay_off(self.under_price)
        if self.next_up is None and self.next_mid is None:
            self.exercised = payoff > 0
        else:
            self.exercised = can_exercise and payoff > hold + 1e-10
        self.option_price = payoff if self.exercised else hold

    # l'option est exercée sur ce noeud (calculé pendant le pricing)
    def is_exercised(self) -> bool:
        return self.exercised

    # AUTRE MÉTHODE POUR CALCULER LE PRIX DE L'OPTION AU DERNIER NOEUD
    # date N-1 : il reste un seul pas avant la maturité, on prend le prix Black-Scholes sur delta_t
    # (spot = prix du noeud moins le dividende du dernier pas s'il y en a un)
    # pour une américaine on garde le max avec la valeur intrinsèque
    def calc_bs_price(self, opt: OptionTrade, trunk: TrunkNode):
        spot: float = self.under_price - self.dividend_pv(trunk)
        mkt: Market = Market(spot, self.tree.mkt.volatility, self.tree.mkt.int_rate, 0.0, None)
        # date de pricing un pas avant la maturité : Black-Scholes sur delta_t
        start: datetime.datetime = opt.mat_date - datetime.timedelta(days=self.tree.delta_t * 365)
        self.set_value(opt, BlackScholes(mkt, start, opt).price(), opt.is_american())

    # valeur actuelle du dividende payé pendant le pas suivant : 0 dans l'arbre binomial (pas de dividende)
    def dividend_pv(self, trunk: TrunkNode) -> float:
        return 0.0
