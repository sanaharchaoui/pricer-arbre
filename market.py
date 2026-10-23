import datetime as datetime


class Market(object):

    def __init__(self, price: float, vol: float, rate: float,
                 div_a: float, ex_date: datetime.datetime):
        self.spot_price: float = price
        self.volatility: float = vol
        self.int_rate: float = rate
        self.div_amount: float = div_a
        self.div_ex_date: datetime.datetime = ex_date
        # pas de prix, de vol ou de dividende négatif
        self.definition_ok = price >= 0 and vol >= 0 and div_a >= 0
