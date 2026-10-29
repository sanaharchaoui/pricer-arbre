import datetime as datetime


class Market(object):

    def __init__(self, price: float, vol: float, rate: float,
                 div_a: float, ex_date: datetime.datetime):
        self.spot_price: float = price
        self.volatility: float = vol
        self.int_rate: float = rate
        self.div_amount: float = div_a
        self.div_ex_date: datetime.datetime = ex_date
        # prix et vol strictement positifs (sinon l'arbre divise par zéro), dividende positif ou nul
        self.definition_ok = price > 0 and vol > 0 and div_a >= 0

    # True si un dividende est payé entre start (exclu) et end (inclus)
    def has_dividend(self, start: datetime.datetime, end: datetime.datetime) -> bool:
        return self.div_amount > 0 and start < self.div_ex_date <= end
