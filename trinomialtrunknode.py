from __future__ import annotations
import datetime as datetime
import math
from trinomialnode import TrinomialNode


# noeud du tronc trinomial : lien vers l'arrière, date de sa colonne et dividende de la période suivante
class TrinomialTrunkNode(TrinomialNode):

    def __init__(self, price: float, tree: TrinomialTree, prec: TrinomialTrunkNode,
                 col_date: datetime.datetime):
        TrinomialNode.__init__(self, price, tree)
        # noeud du tronc de la date précédente, pour revenir en arrière pendant le pricing
        self.prec_trunk: TrinomialTrunkNode = prec
        self.column_date: datetime.datetime = col_date
        # dividende de la période suivante, sa capitalisation et la partie de la période après l'ex-date
        self.div: float = 0.0
        self.div_cap: float = 0.0
        self.div_frac: float = 0.0
        if self.is_div_in_period():
            self.set_dividend()

    def next_date(self) -> datetime.datetime:
        return self.column_date + datetime.timedelta(days=self.tree.delta_t * 365)

    # égalité avec une tolérance
    def are_same_dates(self, d1: datetime.datetime, d2: datetime.datetime) -> bool:
        return abs(d1 - d2) < datetime.timedelta(days=1) / self.tree.nb_steps / 1000

    # le dividende tombe-t-il dans ce pas ?
    # True si date de la colonne < date ex-div <= date suivante
    def is_div_in_period(self) -> bool:
        ex_date: datetime.datetime = self.tree.mkt.div_ex_date
        return self.tree.mkt.div_amount > 0 and not self.are_same_dates(self.column_date, ex_date) \
            and self.column_date < ex_date \
            and (ex_date < self.next_date() or self.are_same_dates(ex_date, self.next_date()))

    # after_ex = t + delta_t - t_ex, le temps entre la date ex-div et la date suivante, en années
    def set_dividend(self):

        # temps entre ex-date et date suivante en années
        after_ex: float = (self.next_date() - self.tree.mkt.div_ex_date).total_seconds() / 86400 / 365
        # montant du dividende
        self.div = self.tree.mkt.div_amount
        # facteur de capitalisation du div de l'ex-date à date suivante
        self.div_cap = math.exp(self.tree.mkt.int_rate * after_ex)
        # part de la période qui tombe après l'ex-date
        self.div_frac = after_ex / self.tree.delta_t
