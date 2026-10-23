import datetime as datetime


class Model(object):

    def __init__(self, pricing_date: datetime.datetime, steps: int, prune_proba: float = 0.0):
        self.price_date: datetime.datetime = pricing_date
        self.steps: int = steps
        # pruning trinomial : un noeud atteint avec une proba plus petite va seulement vers son noeud milieu
        # 0 veut dire pas de pruning
        self.prune_proba: float = prune_proba
