import datetime as datetime


# l'option : maturité, call ou put, européenne ou américaine, strike
class OptionTrade(object):
    # textes possibles pour le type et l'exercice
    CALL_LABEL: str = 'Call'
    PUT_LABEL: str = 'Put'
    AMER_LABEL: str = 'américain'
    EURO_LABEL: str = 'européen'

    def __init__(self, mat: datetime.datetime, call_put: str, ex: str, k: float):
        self.mat_date: datetime.datetime = mat
        self.opt_type: str = call_put
        self.exercice: str = ex
        self.strike: float = k
        self.definition_ok = (self.is_a_call() or self.is_a_put()) \
            and (self.is_european() or self.is_american()) and k >= 0

    # texte affiché par print(option)
    def __str__(self) -> str:
        return self.opt_type + ' ' + self.exercice + ' ' + self.mat_date.strftime("%d-%m-%Y") \
            + ' @' + str(self.strike)

    # True si l'option est américaine (même idée pour les trois méthodes suivantes)
    def is_american(self) -> bool:
        return self.exercice == OptionTrade.AMER_LABEL

    def is_european(self) -> bool:
        return self.exercice == OptionTrade.EURO_LABEL

    def is_a_call(self) -> bool:
        return self.opt_type == OptionTrade.CALL_LABEL

    def is_a_put(self) -> bool:
        return self.opt_type == OptionTrade.PUT_LABEL

    # valeur intrinsèque : max(S - K, 0) pour un call, max(K - S, 0) pour un put
    def pay_off(self, spot: float) -> float:
        if self.is_a_call():
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)
