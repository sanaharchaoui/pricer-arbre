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
    # hypothèse du projet (le cours n'en parle pas) : sinon le prix après dividende serait négatif
    # limite : avec un gros dividende, l'espérance de S n'est plus S - dividende actualisé
    # (le call avec K = 0 vaut alors un peu plus que S - D actualisé)
    def dividend(self, trunk: TrinomialTrunkNode) -> float:
        if self.forward() <= trunk.div * trunk.div_cap:
            return 0.0
        return trunk.div

    # valeur actuelle du dividende payé pendant le pas suivant : D actualisé de l'ex-date à aujourd'hui
    # (div_cap capitalise jusqu'à la date suivante, df actualise d'un pas)
    def dividend_pv(self, trunk: TrinomialTrunkNode) -> float:
        return self.dividend(trunk) * trunk.div_cap * self.tree.df

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

    # valeur si on garde l'option : moyenne actualisée des noeuds suivants (seulement next_mid si pruné)
    def hold_value(self) -> float:
        value: float = self.p_mid * self.next_mid.option_price
        if self.next_up is not None:
            value = value + self.p_up * self.next_up.option_price + self.p_down * self.next_down.option_price
        return self.tree.df * value

    # dernière date : payoff, sinon valeur si on garde (ou le payoff si on exerce une américaine)
    # un noeud pruné n'a qu'un noeud suivant, sa valeur "garder" est approchée :
    # on garde le max avec le payoff, mais on ne le compte pas comme exercé
    def calc_option_price(self, opt: OptionTrade):
        if self.next_mid is None:
            self.set_value(opt, opt.pay_off(self.under_price), True)
        elif self.next_up is None:
            self.set_value(opt, self.hold_value(), False)
            if opt.is_american():
                self.option_price = max(self.option_price, opt.pay_off(self.under_price))
        else:
            self.set_value(opt, self.hold_value(), opt.is_american())
