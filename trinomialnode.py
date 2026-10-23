from __future__ import annotations
import math
from node import Node
from optiontrade import OptionTrade


# noeud de l'arbre trinomial : trois noeuds suivants et ses propres probas
class TrinomialNode(Node):

    def __init__(self, price: float, tree: TrinomialTree):
        Node.__init__(self, price, tree)

        # probas d'aller vers next_up, next_mid et next_down
        self.p_up: float = 0.0
        self.p_mid: float = 0.0
        self.p_down: float = 0.0
        # proba d'atteindre ce noeud depuis la racine (pour le pruning)
        self.proba: float = 0.0

    # dividende payé sur ce noeud : 0 sur les noeuds improbables où l'action vaut moins que le dividende
    def dividend(self, trunk: TrinomialTrunkNode) -> float:
        if self.forward() <= trunk.div * trunk.div_cap:
            return 0.0
        return trunk.div

    # espérance à la date d'après
    def expectation(self, trunk: TrinomialTrunkNode) -> float:
        return self.forward() - self.dividend(trunk) * trunk.div_cap

    # variance à la date d'après
    def variance(self, trunk: TrinomialTrunkNode) -> float:
        t: TrinomialTree = self.tree
        s: float = self.under_price - trunk.div_frac * self.dividend(trunk)
        return s ** 2 * math.exp(2 * t.mkt.int_rate * t.delta_t) \
            * (math.exp(t.mkt.volatility ** 2 * t.delta_t) - 1)

    # probas qui respectent l'espérance E et la variance V
    def calc_probabilities(self, e: float, v: float):
        alpha: float = self.tree.alpha
        s_mid: float = self.next_mid.under_price
        self.p_down = ((v + e ** 2) / s_mid ** 2 - 1 - (alpha + 1) * (e / s_mid - 1)) \
            / ((1 - alpha) * (alpha ** -2 - 1))
        self.p_up = (e / s_mid - 1 - (1 / alpha - 1) * self.p_down) / (alpha - 1)
        self.p_mid = 1 - self.p_up - self.p_down

    # on atteint les noeuds suivants avec la proba de ce noeud x la proba de transition
    def spread_proba(self):
        self.next_mid.proba = self.next_mid.proba + self.proba * self.p_mid
        if self.next_up is not None:
            self.next_up.proba = self.next_up.proba + self.proba * self.p_up
            self.next_down.proba = self.next_down.proba + self.proba * self.p_down

    # l'option est exercée sur ce noeud : à maturité si le payoff est positif,
    # avant la maturité seulement pour une américaine, quand sa valeur est le payoff
    def is_exercised(self, opt: OptionTrade) -> bool:
        payoff: float = opt.pay_off(self.under_price)
        if self.next_mid is None:
            return payoff > 0
        return opt.is_american() and payoff > 0 and self.option_price == payoff

    # dernière date : payoff, sinon moyenne actualisée des noeuds suivants (seulement next_mid si pruné)
    def calc_option_price(self, opt: OptionTrade):
        if self.next_mid is None:
            self.option_price = opt.pay_off(self.under_price)
        else:
            value: float = self.p_mid * self.next_mid.option_price
            if self.next_up is not None:
                value = value + self.p_up * self.next_up.option_price \
                    + self.p_down * self.next_down.option_price
            value = self.tree.df * value
            if opt.is_american():
                value = max(value, opt.pay_off(self.under_price))
            self.option_price = value
