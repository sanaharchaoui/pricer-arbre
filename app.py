import datetime as datetime
import streamlit as st
from matplotlib.figure import Figure
from market import Market
from model import Model
from optiontrade import OptionTrade
from trinomialtree import TrinomialTree
from blackscholes import BlackScholes
from greeks import Greeks
from treeimage import TreeImage
from study import Study


# interface du pricer : python -m streamlit run app.py
# seule run() est publique, tout le reste est privé (méthodes qui commencent par _)
class PricerApp(object):
    PRICING_DATE: datetime.datetime = datetime.datetime(2026, 9, 15)
    PRUNE: float = 1e-7

    # les graphes d'études : nom -> fonction qui prend l'étude et renvoie la figure
    STUDIES: dict = {
        "01 - Écart x nombre de pas": lambda s: PricerApp._with_convergence(s).convergence_gap_graph(),
        "02 - Écart selon le strike (10 pas)": lambda s: s.strike_gap_graph(10, 0.8 * s.opt.strike,
                                                                             1.2 * s.opt.strike, 0.1),
        "03 - Américaine - européenne selon le taux": lambda s: s.rate_graph(100, -0.05, 0.05, 0.01),
        "05 - Temps avec et sans pruning": lambda s: s.pruning_graph([10, 20, 50, 100, 200, 300, 500],
                                                                     PricerApp.PRUNE),
        "06 - Dernier pas : écart x nombre de pas":
            lambda s: s.last_step_graph([1, 2, 3, 5, 10, 20, 50, 100, 200, 300]),
        "07 - Prix selon le nombre de pas":
            lambda s: PricerApp._with_convergence(s).convergence_price_graph(),
        "08 - Probabilité d'exercice par date": lambda s: s.exercise_graph(20),
        "09 - Prix selon la date ex-dividende": lambda s: s.ex_date_graph(100, 14),
        "10 - Tronc avec dividende": lambda s: s.trunk_graph(100),
        "11 - Dernier pas : écart selon le strike":
            lambda s: s.last_step_strike_graph(10, 0.7 * s.opt.strike, 1.3 * s.opt.strike, 0.25),
        "12 - Delta selon le nombre de pas": lambda s: s.delta_graph(list(range(10, 201, 10))),
    }

    def __init__(self):
        # paramètres choisis dans la barre latérale, lus par run()
        self.params: tuple = ()
        self.use_bs: bool = False

    # la page : barre latérale, prix, grecques, puis les trois onglets
    def run(self):
        st.set_page_config(page_title="Pricer arbre trinomial", layout="wide")
        st.title("Pricer d'options par arbre trinomial")
        self.params = self._market_sidebar() + self._option_sidebar()
        self.use_bs = st.sidebar.checkbox("Black-Scholes sur le dernier pas", value=False)
        self._show_prices()
        self._show_greeks()
        tab_tree, tab_boundary, tab_studies = st.tabs(["Arbre", "Frontière d'exercice", "Études"])
        with tab_tree:
            self._show_tree()
        with tab_boundary:
            self._show_boundary()
        with tab_studies:
            self._show_studies()

    # ==================== calculs ====================
    # gardés en cache : un même jeu de paramètres n'est calculé qu'une fois

    # marché avec dividende, modèle et option à partir des paramètres choisis dans la barre latérale
    @staticmethod
    def _objects(p: tuple) -> tuple:
        spot, vol, rate, div, ex_days, mat_days, strike, call_put, ex, steps, prune = p
        date: datetime.datetime = PricerApp.PRICING_DATE
        mkt: Market = Market(spot, vol, rate, div, date + datetime.timedelta(days=ex_days))
        mdl: Model = Model(date, steps, prune)
        opt: OptionTrade = OptionTrade(date + datetime.timedelta(days=mat_days), call_put, ex, strike)
        return mkt, mdl, opt

    # prix de l'arbre avec et sans dividende, et prix Black-Scholes (None pour une américaine)
    @staticmethod
    @st.cache_data
    def _prices(p: tuple, last_step_bs: bool) -> tuple:
        mkt, mdl, opt = PricerApp._objects(p)
        study: Study = Study(mkt, mdl, opt)
        price: float = study.tree(mdl.steps, True, mkt, mdl.prune_proba).price_option(opt, last_step_bs)
        tree0: TrinomialTree = study.tree(mdl.steps, True, study.mkt0, mdl.prune_proba)
        price0: float = tree0.price_option(opt, last_step_bs)
        bs: float = study.bs_price(opt) if opt.is_european() else None
        return price, price0, bs

    # delta, gamma et vega (pour 1 % de vol) de l'arbre
    @staticmethod
    @st.cache_data
    def _greeks(p: tuple) -> tuple:
        mkt, mdl, opt = PricerApp._objects(p)
        g: Greeks = Greeks(mkt, mdl, opt)
        return g.delta(), g.gamma(), g.vega() * 0.01

    # grecques Black-Scholes, seulement si elles sont comparables : option européenne et pas de dividende
    @staticmethod
    @st.cache_data
    def _bs_greeks(p: tuple) -> tuple:
        mkt, mdl, opt = PricerApp._objects(p)
        if not (opt.is_european() and mkt.div_amount == 0.0):
            return None
        bs: BlackScholes = BlackScholes(mkt, PricerApp.PRICING_DATE, opt)
        return bs.delta(), bs.gamma(), bs.vega() * 0.01

    # image svg de l'arbre trinomial, après le pricing de l'option
    @staticmethod
    @st.cache_data
    def _tree_svg(p: tuple) -> str:
        mkt, mdl, opt = PricerApp._objects(p)
        tree: TrinomialTree = TrinomialTree(mkt, mdl, opt.mat_date)
        tree.build_tree()
        tree.price_option(opt)
        return TreeImage(1000, 600, tree, opt).image.as_str()

    # convergences binomiale et trinomiale jusqu'à 150 pas, utilisées par les graphes 1 et 7
    @staticmethod
    def _with_convergence(study: Study) -> Study:
        study.convergence("binomial", 150, False)
        study.convergence("trinomial", 150, True)
        return study

    # graphe d'étude choisi, sur le marché choisi, avec un call européen de même strike et même maturité
    @staticmethod
    @st.cache_resource
    def _study_figure(p: tuple, name: str) -> Figure:
        mkt, mdl, opt = PricerApp._objects(p)
        call: OptionTrade = OptionTrade(opt.mat_date, OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL,
                                        opt.strike)
        return PricerApp.STUDIES[name](Study(mkt, mdl, call))

    # frontière d'exercice du put américain, sans et avec dividende
    @staticmethod
    @st.cache_resource
    def _boundary_figure(p: tuple) -> Figure:
        mkt, mdl, opt = PricerApp._objects(p)
        return Study(mkt, mdl, opt).exercise_boundary_graph(mdl.steps, PricerApp.PRUNE)

    # ==================== affichage ====================

    # barre latérale : paramètres du marché
    # (le dividende est détaché ex_days jours après la date de pricing)
    def _market_sidebar(self) -> tuple:
        st.sidebar.header("Marché")
        spot: float = st.sidebar.number_input("Spot S0", value=100.0, min_value=1.0, step=1.0)
        vol: float = st.sidebar.slider("Volatilité σ", 0.05, 0.80, 0.20, 0.01)
        rate: float = st.sidebar.slider("Taux d'intérêt r", -0.05, 0.15, 0.03, 0.005)
        div: float = st.sidebar.number_input("Dividende (€)", value=3.0, min_value=0.0, step=0.5)
        ex_days: int = st.sidebar.slider("Date ex-dividende (jours après le pricing)", 1, 720, 140)
        return spot, vol, rate, div, ex_days

    # barre latérale : paramètres de l'option et de l'arbre
    def _option_sidebar(self) -> tuple:
        st.sidebar.header("Option et arbre")
        mat_days: int = st.sidebar.slider("Maturité (jours)", 30, 730, 352)
        strike: float = st.sidebar.number_input("Strike K", value=101.0, min_value=1.0, step=1.0)
        call_put: str = st.sidebar.selectbox("Type", [OptionTrade.CALL_LABEL, OptionTrade.PUT_LABEL])
        ex: str = st.sidebar.selectbox("Exercice", [OptionTrade.EURO_LABEL, OptionTrade.AMER_LABEL])
        steps: int = st.sidebar.slider("Nombre de pas N", 10, 500, 100, 10)
        prune: float = st.sidebar.selectbox("Seuil de pruning", [0.0, 1e-7, 1e-6], index=1)
        return mat_days, strike, call_put, ex, steps, prune

    # prix de l'arbre, prix sans dividende et comparaison avec Black-Scholes
    def _show_prices(self):
        price, price0, bs = PricerApp._prices(self.params, self.use_bs)
        col1, col2, col3 = st.columns(3)
        col1.metric("Prix arbre trinomial", f"{price:.4f} €")
        col2.metric("Arbre sans dividende", f"{price0:.4f} €")
        if bs is None:
            col3.metric("Black-Scholes", "européenne seulement")
        else:
            col3.metric("Black-Scholes (sans dividende)", f"{bs:.4f} €",
                        f"écart arbre : {price0 - bs:+.5f}", delta_color="off")

    # grecques de l'arbre, avec Black-Scholes en dessous quand c'est comparable
    def _show_greeks(self):
        values: tuple = PricerApp._greeks(self.params)
        refs: tuple = PricerApp._bs_greeks(self.params)
        labels: list = ["Delta", "Gamma", "Vega (1 % de vol)"]
        for i, col in enumerate(st.columns(3)):
            ref: str = "" if refs is None else f"BS : {refs[i]:.5f}"
            col.metric(labels[i], f"{values[i]:.5f}", ref, delta_color="off")
        if refs is None:
            st.caption("Black-Scholes : comparaison seulement pour une européenne sans dividende.")

    # onglet arbre : l'image n'est lisible que pour un petit nombre de pas
    def _show_tree(self):
        if self.params[9] > 30:
            st.info("Mettez N ≤ 30 pas pour afficher l'arbre.")
        else:
            st.image(PricerApp._tree_svg(self.params))
            st.caption("Couleur : valeur de l'option (blanc = 0). Bord rouge : exercice anticipé.")

    # onglet frontière d'exercice
    def _show_boundary(self):
        st.pyplot(PricerApp._boundary_figure(self.params))
        st.caption("Au-dessus de la courbe on garde le put américain, en dessous on l'exerce.")

    # onglet études : l'utilisateur choisit le graphe à afficher
    def _show_studies(self):
        name: str = st.selectbox("Graphe", list(PricerApp.STUDIES))
        st.caption("Études faites avec un call européen de même strike et même maturité,"
                   " sur le marché choisi.")
        with st.spinner("Calcul en cours (une seule fois par jeu de paramètres)..."):
            st.pyplot(PricerApp._study_figure(self.params, name))


PricerApp().run()
