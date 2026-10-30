from __future__ import annotations

import datetime as datetime
import math
from market import Market
from optiontrade import OptionTrade


class BlackScholes(object):

    def __init__(self, mkt: Market, pricing_date: datetime.datetime, opt: OptionTrade):
        self.mkt: Market = mkt
        self.price_date: datetime.datetime = pricing_date
        self.opt: OptionTrade = opt
        # temps jusqu'à la maturité, en années
        self.mat_time: float = (opt.mat_date - pricing_date).total_seconds() / 86400 / 365
        # B-S marche seulement pour une option européenne sans dividende
        self.definition_ok = opt.is_european() and (mkt.div_amount == 0.0) \
                             and (0.0 < opt.strike) and (0.0 < mkt.volatility) and (0.0 < self.mat_time)

    # fonction de répartition de la loi normale
    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # densité de la loi normale
    def norm_pdf(self, x: float) -> float:
        return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)

    def d1(self) -> float:
        vol: float = self.mkt.volatility
        t: float = self.mat_time
        return (math.log(self.mkt.spot_price / self.opt.strike) + (self.mkt.int_rate + vol ** 2 / 2) * t) \
            / (vol * math.sqrt(t))

    def d2(self) -> float:
        return self.d1() - self.mkt.volatility * math.sqrt(self.mat_time)

    # prix Black-Scholes du call ou du put européen
    def price(self) -> float:
        s: float = self.mkt.spot_price
        k: float = self.opt.strike
        df: float = math.exp(-self.mkt.int_rate * self.mat_time)
        # strike nul : le call vaut le spot, le put ne vaut rien (on évite log(S / 0))
        if k == 0.0:
            if self.opt.is_a_call():
                return s
            return 0.0
        if self.opt.is_a_call():
            return s * self.norm_cdf(self.d1()) - k * df * self.norm_cdf(self.d2())
        return k * df * self.norm_cdf(-self.d2()) - s * self.norm_cdf(-self.d1())

    # delta : N(d1) pour un call, N(d1) - 1 pour un put
    def delta(self) -> float:
        if self.opt.is_a_call():
            return self.norm_cdf(self.d1())
        return self.norm_cdf(self.d1()) - 1

    # gamma et vega (pour 1 unité de vol) sont les mêmes pour un call et un put
    def gamma(self) -> float:
        s: float = self.mkt.spot_price
        return self.norm_pdf(self.d1()) / (s * self.mkt.volatility * math.sqrt(self.mat_time))

    def vega(self) -> float:
        return self.mkt.spot_price * self.norm_pdf(self.d1()) * math.sqrt(self.mat_time)

    # vomma : dérivée du vega par rapport à la vol (pour 1 unité de vol)
    def vomma(self) -> float:
        return self.vega() * self.d1() * self.d2() / self.mkt.volatility

    # vanna : dérivée du delta par rapport à la vol (pour 1 unité de vol)
    def vanna(self) -> float:
        return -self.norm_pdf(self.d1()) * self.d2() / self.mkt.volatility
