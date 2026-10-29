from __future__ import annotations
import datetime as datetime
from market import Market
from model import Model
from node import Node
from optiontrade import OptionTrade
from trinomialtree import TrinomialTree
from derivatives import OneDimDerivative


# prix de l'arbre en fonction du spot, les autres paramètres sont dans l'objet Greeks
def price_with_spot(greeks: Greeks, spot: float) -> float:
    return greeks.priced_tree(spot, greeks.mkt.volatility, greeks.mdl.price_date).price_node().option_price


# prix de l'arbre en fonction de la vol, les autres paramètres sont dans l'objet Greeks
def price_with_vol(greeks: Greeks, vol: float) -> float:
    return greeks.priced_tree(greeks.mkt.spot_price, vol, greeks.mdl.price_date).price_node().option_price


# delta de l'arbre en fonction de la vol, les autres paramètres sont dans l'objet Greeks
def delta_with_vol(greeks: Greeks, vol: float) -> float:
    return Greeks.tree_delta(greeks.priced_tree(greeks.mkt.spot_price, vol, greeks.mdl.price_date))


# grecques avec l'arbre trinomial :
# - delta et gamma de deux façons : avec les trois noeuds de la date de pricing (arbre qui commence
#   un pas avant), ou avec OneDimDerivative et un choc de 1 % sur le spot
# - vega, vomma, vanna avec OneDimDerivative et un choc de 1 % de vol ; theta en recalculant l'arbre
# dernier pas : le même que celui du prix (last_step_bs) ; avec la valeur intrinsèque, un choc sur
# le spot ou la vol déplace les noeuds par rapport au strike et ce bruit gêne les dérivées secondes,
# Black-Scholes sur le dernier pas l'enlève
class Greeks(object):
    # chocs : 1 % du spot pour delta et gamma, 1 % de vol pour vega, vomma et vanna
    SPOT_SHIFT: float = 0.01
    VOL_SHIFT: float = 0.01

    def __init__(self, mkt: Market, mdl: Model, opt: OptionTrade, last_step_bs: bool = True):
        self.mkt: Market = mkt
        self.mdl: Model = mdl
        self.opt: OptionTrade = opt
        # méthode du dernier pas : Black-Scholes en N-1 (True) ou valeur intrinsèque en N (False)
        self.last_step_bs: bool = last_step_bs
        # arbre avec les valeurs de l'option sur chaque noeud, utilisé par delta et gamma
        self.tree: TrinomialTree = self.priced_tree(mkt.spot_price, mkt.volatility, mdl.price_date)
        self.price: float = self.tree.price_node().option_price

    # arbre trinomial avec un autre spot, une autre vol et une autre date de pricing, construit et pricé
    def priced_tree(self, spot: float, vol: float, price_date: datetime.datetime) -> TrinomialTree:
        m: Market = self.mkt
        mkt: Market = Market(spot, vol, m.int_rate, m.div_amount, m.div_ex_date)
        mdl: Model = Model(price_date, self.mdl.steps, self.mdl.prune_proba)
        tree: TrinomialTree = TrinomialTree(mkt, mdl, self.opt.mat_date, False, True)
        tree.build_tree()
        tree.price_option(self.opt, self.last_step_bs)
        return tree

    # delta : pente entre les noeuds up et down de la date de pricing
    @staticmethod
    def tree_delta(tree: TrinomialTree) -> float:
        up: Node = tree.root.next_up
        down: Node = tree.root.next_down
        return (up.option_price - down.option_price) / (up.under_price - down.under_price)

    def delta(self) -> float:
        return Greeks.tree_delta(self.tree)

    # gamma avec les trois noeuds de la date de pricing : variation du delta entre up-mid et mid-down
    def gamma(self) -> float:
        up: Node = self.tree.root.next_up
        mid: Node = self.tree.root.next_mid
        down: Node = self.tree.root.next_down
        delta_up: float = (up.option_price - mid.option_price) / (up.under_price - mid.under_price)
        delta_down: float = (mid.option_price - down.option_price) / (mid.under_price - down.under_price)
        return (delta_up - delta_down) / ((up.under_price - down.under_price) / 2)

    # delta avec un choc sur le spot : dérivée du prix par rapport au spot, choc de 1 % du spot
    def delta_shift(self) -> float:
        return OneDimDerivative(price_with_spot, self, self.mkt.spot_price * Greeks.SPOT_SHIFT) \
            .first(self.mkt.spot_price)

    # gamma avec un choc sur le spot : dérivée seconde du prix par rapport au spot, choc de 1 % du spot
    def gamma_shift(self) -> float:
        return OneDimDerivative(price_with_spot, self, self.mkt.spot_price * Greeks.SPOT_SHIFT) \
            .second(self.mkt.spot_price)

    # vega : dérivée du prix par rapport à la vol, choc de 1 % de vol
    def vega(self) -> float:
        return OneDimDerivative(price_with_vol, self, Greeks.VOL_SHIFT).first(self.mkt.volatility)

    # vomma : dérivée seconde du prix par rapport à la vol, choc de 1 % de vol
    def vomma(self) -> float:
        return OneDimDerivative(price_with_vol, self, Greeks.VOL_SHIFT).second(self.mkt.volatility)

    # vanna : dérivée du delta par rapport à la vol, choc de 1 % de vol
    def vanna(self) -> float:
        return OneDimDerivative(delta_with_vol, self, Greeks.VOL_SHIFT).first(self.mkt.volatility)

    # theta : prix demain - prix aujourd'hui, même vol, même spot
    # si le dividende tombe demain, le spot de demain est ex-dividende : spot - dividende
    # (sinon l'option gagnerait la chute du dividende, qui n'est pas un effet du temps)
    def theta(self) -> float:
        tomorrow: datetime.datetime = self.mdl.price_date + datetime.timedelta(days=1)
        spot: float = self.mkt.spot_price
        if self.mkt.has_dividend(self.mdl.price_date, tomorrow):
            spot = spot - self.mkt.div_amount
        return self.priced_tree(spot, self.mkt.volatility, tomorrow).price_node().option_price - self.price
