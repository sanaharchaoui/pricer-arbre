from __future__ import annotations
from market import Market
from model import Model
from node import Node
from optiontrade import OptionTrade
from trinomialtree import TrinomialTree
from derivatives import OneDimDerivative


# prix de l'arbre en fonction de la vol, les autres paramètres sont dans l'objet Greeks
def price_with_vol(greeks: object, vol: float) -> float:
    mkt: Market = Market(greeks.mkt.spot_price, vol, greeks.mkt.int_rate, greeks.mkt.div_amount,
                         greeks.mkt.div_ex_date)
    tree: TrinomialTree = TrinomialTree(mkt, greeks.mdl, greeks.opt.mat_date)
    tree.build_tree()
    return tree.price_option(greeks.opt)


class Greeks(object):

    def __init__(self, mkt: Market, mdl: Model, opt: OptionTrade):
        self.mkt: Market = mkt
        self.mdl: Model = mdl
        self.opt: OptionTrade = opt
        # valeur de l'option sur chaque noeud, utilisée par delta et gamma
        self.tree: TrinomialTree = TrinomialTree(mkt, mdl, opt.mat_date)
        self.tree.build_tree()
        self.price: float = self.tree.price_option(opt)

    # delta avec les noeuds up et down de la première date
    def delta(self) -> float:
        up: Node = self.tree.root.next_up
        down: Node = self.tree.root.next_down
        return (up.option_price - down.option_price) / (up.under_price - down.under_price)

    # gamma avec les trois noeuds de la première date : variation du delta entre up-mid et mid-down
    def gamma(self) -> float:
        up: Node = self.tree.root.next_up
        mid: Node = self.tree.root.next_mid
        down: Node = self.tree.root.next_down
        delta_up: float = (up.option_price - mid.option_price) / (up.under_price - mid.under_price)
        delta_down: float = (mid.option_price - down.option_price) / (mid.under_price - down.under_price)
        return (delta_up - delta_down) / ((up.under_price - down.under_price) / 2)

    # vega : dérivée du prix par rapport à la vol, choc de 1 % de vol
    def vega(self) -> float:
        return OneDimDerivative(price_with_vol, self, 0.01).first(self.mkt.volatility)
