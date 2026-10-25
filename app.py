import datetime as datetime
import streamlit as st
from matplotlib.figure import Figure
from market import Market
from model import Model
from optiontrade import OptionTrade
from tree import Tree
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
        # convergence et strike
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
        # exercice américain et dividende
        "08 - Probabilité d'exercice par date": lambda s: s.exercise_graph(20),
        "09 - Prix selon la date ex-dividende": lambda s: s.ex_date_graph(100, 14),
        "10 - Tronc avec dividende": lambda s: s.trunk_graph(100),
        "11 - Dernier pas : écart selon le strike":
            lambda s: s.last_step_strike_graph(10, 0.7 * s.opt.strike, 1.3 * s.opt.strike, 0.25),
        "12 - Delta selon le nombre de pas": lambda s: s.delta_graph(list(range(10, 201, 10))),
    }

    # style de la page : cartes des métriques (bordure, ombre, petit mouvement au survol) et onglets
    CSS: str = """<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 14px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: transform 0.2s ease, box-shadow 0.2s ease; }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        border-color: rgba(0,150,255,0.4); }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    .stTabs [data-baseweb="tab"] { padding: 10px 22px; border-radius: 8px 8px 0 0; font-weight: 600;
        font-size: 0.95rem; }
    </style>"""

    # notes d'analyse affichées au-dessus de chaque graphe d'étude
    STUDY_INSIGHTS: dict = {
        # convergence et strike
        "01 - Écart x nombre de pas":
            "L'erreur décroît en 1/N. Le binomial oscille fortement entre N pair et N impair,"
            " le trinomial converge de façon beaucoup plus régulière.",
        "02 - Écart selon le strike (10 pas)":
            "L'écart forme des arches entre deux noeuds de la maturité : entre deux noeuds, le prix"
            " de l'arbre est linéaire en K alors que Black-Scholes est convexe. L'écart est minimal"
            " sur les noeuds.",
        "03 - Américaine - européenne selon le taux":
            "Sans dividende, la prime d'exercice anticipé est portée par le call quand r < 0 et par"
            " le put quand r > 0 ; à r = 0 les deux prix sont égaux. Le dividende rend le call"
            " américain plus cher.",
        "05 - Temps avec et sans pruning":
            "Le pruning (seuil 1e-7) supprime environ 91 % des noeuds à 1000 pas et divise le temps"
            " de calcul par plus de 10, pour un écart de prix d'environ 1e-6 €.",
        "06 - Dernier pas : écart x nombre de pas":
            "Black-Scholes sur le dernier pas (en N-1) supprime l'oscillation. Avec le trinomial,"
            " l'erreur devient environ 100 fois plus petite qu'avec la valeur intrinsèque.",
        "07 - Prix selon le nombre de pas":
            "Les prix binomial et trinomial convergent vers Black-Scholes ; le binomial en oscillant,"
            " le trinomial beaucoup plus régulièrement.",
        # exercice américain et dividende
        "08 - Probabilité d'exercice par date":
            "Avec nos paramètres, le put américain avec dividende n'est jamais exercé avant la date"
            " ex-dividende, et le call américain avec dividende l'est juste avant cette date.",
        "09 - Prix selon la date ex-dividende":
            "Plus le dividende est détaché tard, plus le call américain vaut cher par rapport au call"
            " européen : on peut exercer juste avant le détachement.",
        "10 - Tronc avec dividende":
            "Le tronc (le forward) chute du montant du dividende capitalisé à la date ex-dividende.",
        # dernier pas et grecques
        "11 - Dernier pas : écart selon le strike":
            "Black-Scholes en N-1 supprime les arches de l'écart selon le strike : la courbe du"
            " trinomial devient presque plate.",
        "12 - Delta selon le nombre de pas":
            "Le delta lu sur les noeuds de la première date converge vers le delta de Black-Scholes"
            " quand N augmente.",
    }

    def __init__(self):
        # paramètres choisis dans la barre latérale, lus par run()
        self.params: tuple = ()
        self.trinomial: bool = True
        self.use_bs: bool = False

    # la page : thème, en-tête, barre latérale, prix, grecques, puis les trois onglets
    def run(self):
        st.set_page_config(page_title="Pricer Arbres Binomial & Trinomial", page_icon="📊", layout="wide")
        self._inject_custom_theme()
        self._render_header()
        self.params = self._market_sidebar() + self._option_sidebar() + self._tree_sidebar()
        self._show_prices()
        self._show_greeks()
        self._show_tabs()

    # les trois onglets : arbre, frontière d'exercice, études
    def _show_tabs(self):
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        tabs: list = st.tabs(["🌳 Visualisation de l'Arbre", "🎯 Frontière d'Exercice S*(t)",
                              "📈 Études Numériques & Stabilité"])
        with tabs[0]:
            self._show_tree()
        with tabs[1]:
            self._show_boundary()
        with tabs[2]:
            self._show_studies()

    # ==================== mise en page ====================

    # CSS : cartes de métriques avec bordure et ombre légères, onglets plus lisibles
    @staticmethod
    def _inject_custom_theme():
        st.markdown(PricerApp.CSS, unsafe_allow_html=True)

    # titre et sous-titre de la page
    @staticmethod
    def _render_header():
        st.title("📈 Pricer d'Options par Arbres Binomial & Trinomial")
        st.caption("Pricing backward d'options européennes et américaines, avec dividende discret,"
                   " pruning et monitoring des probabilités.")

    # ==================== barre latérale ====================

    # paramètres du marché, avec le nom du projet en haut (texte simple, sans badge)
    def _market_sidebar(self) -> tuple:
        st.sidebar.markdown("**Projet M2 Quant Dauphine**  \nArbres binomial et trinomial · pruning 1e-7")
        st.sidebar.markdown("---\n### 🏛️ Paramètres de Marché")
        spot: float = st.sidebar.number_input("Spot S0 (€)", value=100.0, min_value=1.0, step=1.0,
                                              help="Prix actuel du sous-jacent.")
        vol: float = st.sidebar.slider("Volatilité σ", 0.05, 0.80, 0.20, 0.01,
                                       help="Volatilité constante du modèle (annualisée).")
        rate: float = st.sidebar.slider("Taux d'intérêt sans risque r", -0.05, 0.15, 0.03, 0.005,
                                        help="Taux d'intérêt continu sans risque.")
        div: float = st.sidebar.number_input("Dividende discret (€)", value=3.0, min_value=0.0,
                                             step=0.5, help="Montant versé à la date ex-dividende.")
        ex_days: int = st.sidebar.slider("Détachement ex-dividende (jours)", 1, 720, 140,
                                         help="Nombre de jours entre la date de pricing et l'ex-date.")
        return spot, vol, rate, div, ex_days

    # paramètres de l'option et de l'arbre
    def _option_sidebar(self) -> tuple:
        st.sidebar.markdown("### 📄 Contrat de l'Option")
        mat_days: int = st.sidebar.slider("Maturité T (jours)", 30, 730, 352,
                                          help="Durée totale jusqu'à l'expiration du contrat.")
        strike: float = st.sidebar.number_input("Prix d'exercice K (€)", value=101.0, min_value=1.0,
                                                step=1.0, help="Prix d'achat (call) ou de vente (put).")
        call_put: str = st.sidebar.selectbox("Type de dérivé",
                                             [OptionTrade.CALL_LABEL, OptionTrade.PUT_LABEL])
        ex: str = st.sidebar.selectbox("Style d'exercice", [OptionTrade.EURO_LABEL, OptionTrade.AMER_LABEL],
                                       help="Européen : exercice à maturité uniquement."
                                            " Américain : exercice possible à tout moment.")
        return mat_days, strike, call_put, ex

    # modèle : type d'arbre, méthode du dernier pas, nombre de pas et pruning
    def _tree_sidebar(self) -> tuple:
        st.sidebar.markdown("### ⚙️ Modèle Numérique")
        tree_type: str = st.sidebar.radio("Architecture de l'arbre", ["trinomial", "binomial"],
                                          horizontal=True,
                                          help="Seul le trinomial prend en compte le dividende discret.")
        self.trinomial = tree_type == "trinomial"
        self.use_bs = st.sidebar.checkbox("Black-Scholes sur le dernier pas (N-1)", value=False,
                                          help="Black-Scholes sur delta_t au lieu de la valeur intrinsèque.")
        steps: int = st.sidebar.slider("Nombre de pas temporels N", 10, 500, 100, 10,
                                       help="Finesse de la discrétisation en temps de l'arbre.")
        prune: float = st.sidebar.selectbox("Seuil d'élagage (Pruning)", [0.0, 1e-7, 1e-6], index=1,
                                            help="Sous ce seuil de probabilité, un noeud n'a qu'un suivant.")
        return steps, prune

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
    # (l'arbre binomial ne prend pas en compte le dividende : ses deux prix sont égaux)
    @staticmethod
    @st.cache_data
    def _prices(p: tuple, trinomial: bool, last_step_bs: bool) -> tuple:
        mkt, mdl, opt = PricerApp._objects(p)
        study: Study = Study(mkt, mdl, opt)
        price: float = study.tree(mdl.steps, trinomial, mkt, mdl.prune_proba).price_option(opt, last_step_bs)
        tree0: Tree = study.tree(mdl.steps, trinomial, study.mkt0, mdl.prune_proba)
        price0: float = tree0.price_option(opt, last_step_bs)
        bs: float = study.bs_price(opt) if opt.is_european() else None
        return price, price0, bs

    # delta, gamma et vega (pour 1 % de vol) de l'arbre trinomial
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

    # image svg de l'arbre binomial ou trinomial, après le pricing de l'option
    @staticmethod
    @st.cache_data
    def _tree_svg(p: tuple, trinomial: bool) -> str:
        mkt, mdl, opt = PricerApp._objects(p)
        tree: Tree = Study(mkt, mdl, opt).tree(mdl.steps, trinomial, mkt, mdl.prune_proba)
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

    # prix de l'arbre, prix sans dividende et comparaison avec Black-Scholes
    def _show_prices(self):
        price, price0, bs = PricerApp._prices(self.params, self.trinomial, self.use_bs)
        if not self.trinomial and self.params[3] > 0:
            st.warning("⚠️ L'arbre binomial ne prend pas en compte le dividende"
                       " (seul le trinomial le fait).")
        st.markdown("#### 💰 Valorisation du Contrat")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prix Arbre " + ("Trinomial" if self.trinomial else "Binomial"), f"{price:.4f} €",
                    help="Prix obtenu par pricing backward sur l'arbre.")
        col2.metric("Arbre (Marché sans dividende)", f"{price0:.4f} €",
                    help="Prix sur le même arbre, sans dividende.")
        self._show_bs(col3, price0, bs)

    # carte Black-Scholes : écart avec l'arbre sans dividende, ou N/A pour une américaine
    def _show_bs(self, col: object, price0: float, bs: float):
        if bs is None:
            col.metric("Black-Scholes Analytique", "N/A (américaine)",
                       help="Black-Scholes n'a pas de formule fermée pour les options américaines.")
        else:
            col.metric("Black-Scholes (sans dividende)", f"{bs:.4f} €", f"Écart : {price0 - bs:+.5f} €",
                       delta_color="off", help="Référence exacte pour une européenne sans dividende.")

    # grecques de l'arbre, avec Black-Scholes en dessous quand c'est comparable
    def _show_greeks(self):
        st.markdown("#### ⚡ Sensibilités & Couverture (Grecques)")
        values: tuple = PricerApp._greeks(self.params)
        refs: tuple = PricerApp._bs_greeks(self.params)
        labels: list = ["Delta (Δ)", "Gamma (Γ)", "Vega (ν, choc 1% vol)"]
        tooltips: list = ["Ratio de couverture : variation du prix de l'option pour +1 € sur le spot.",
                          "Convexité : variation du delta pour +1 € sur le spot.",
                          "Variation du prix de l'option pour une hausse de 1 point de volatilité."]
        for i, col in enumerate(st.columns(3)):
            ref: str = "" if refs is None else f"Réf. BS : {refs[i]:.5f}"
            col.metric(labels[i], f"{values[i]:.5f}", ref, delta_color="off", help=tooltips[i])
        st.caption("ℹ️ Grecques calculées avec l'arbre trinomial. Référence Black-Scholes affichée"
                   " seulement pour une européenne sans dividende.")

    # onglet arbre : l'image n'est lisible que pour un petit nombre de pas
    def _show_tree(self):
        if self.params[9] > 30:
            st.info("💡 **Conseil de lisibilité** : sélectionnez **N ≤ 30** dans la barre latérale"
                    " pour afficher l'arbre.")
        else:
            st.image(PricerApp._tree_svg(self.params, self.trinomial))
            st.caption("🎨 **Légende** : blanc = valeur nulle | du bleu au vert = valeur croissante"
                       " | bord rouge = exercice anticipé.")

    # onglet frontière d'exercice, avec son explication
    def _show_boundary(self):
        st.pyplot(PricerApp._boundary_figure(self.params))
        with st.expander("💡 Comprendre la frontière d'exercice S*(t)", expanded=True):
            st.markdown("""
* **Règle d'exercice** : à chaque date $t$, si le spot est sous le prix critique $S^*(t)$, la valeur
  d'exercice immédiat ($K - S$) dépasse la valeur de continuation : **on exerce le put**.
* **Effet du dividende discret** : avec nos paramètres, le put américain **n'est jamais exercé avant la
  date ex-dividende**. Le détenteur attend la baisse du cours de $D$ euros au détachement. Après
  l'ex-date, la frontière redevient celle du put sans dividende.
* **À l'échéance** : la valeur temps disparaît et la frontière rejoint le strike : $S^*(t) \\to K$
  quand $t \\to T$.
            """)

    # onglet études : l'utilisateur choisit le graphe, avec une note sur le résultat
    def _show_studies(self):
        name: str = st.selectbox("Sélectionnez l'étude à afficher :", list(PricerApp.STUDIES))
        if name in PricerApp.STUDY_INSIGHTS:
            st.info(f"📌 **Constat** : {PricerApp.STUDY_INSIGHTS[name]}")
        st.caption("Études faites avec un call européen de même strike et même maturité,"
                   " sur le marché choisi.")
        with st.spinner("Calcul en cours (une seule fois par jeu de paramètres)..."):
            st.pyplot(PricerApp._study_figure(self.params, name))


PricerApp().run()
