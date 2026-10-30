import datetime as datetime
import streamlit as st
from market import Market
from model import Model
from optiontrade import OptionTrade
from blackscholes import BlackScholes
from greeks import Greeks
from treeimage import TreeImage
from study import Study


# application du pricer : python -m streamlit run app.py
# streamlit relance tout le fichier à chaque changement d'un paramètre


# une date du calendrier (datetime.date) en datetime.datetime à minuit, comme dans le reste du projet
def to_datetime(d: datetime.date) -> datetime.datetime:
    return datetime.datetime(d.year, d.month, d.day)


st.title("Pricer d'options par arbres binomial et trinomial")

# =============== PARAMÈTRES (barre de gauche) ===============
st.sidebar.header("Marché")
spot = st.sidebar.number_input("Spot", value=100.0, min_value=1.0, step=1.0)
vol = st.sidebar.number_input("Volatilité", value=0.20, min_value=0.01, step=0.01)
rate = st.sidebar.number_input("Taux d'intérêt", value=0.03, step=0.005, format="%.3f")
div = st.sidebar.number_input("Dividende", value=3.0, min_value=0.0, step=0.5)

st.sidebar.header("Dates")
pricing_date = st.sidebar.date_input("Date de pricing", value=datetime.date(2026, 9, 15))
ex_date = st.sidebar.date_input("Date ex-dividende", value=datetime.date(2027, 2, 2))
mat_date = st.sidebar.date_input("Maturité", value=datetime.date(2027, 9, 2))
if mat_date <= pricing_date:
    st.error("La maturité doit être après la date de pricing.")
    st.stop()

st.sidebar.header("Option")
strike = st.sidebar.number_input("Strike", value=101.0, min_value=1.0, step=1.0)
call_put = st.sidebar.selectbox("Type", [OptionTrade.CALL_LABEL, OptionTrade.PUT_LABEL])
exercise = st.sidebar.selectbox("Exercice", [OptionTrade.EURO_LABEL, OptionTrade.AMER_LABEL])

st.sidebar.header("Arbre")
trinomial = st.sidebar.radio("Arbre", ["trinomial", "binomial"]) == "trinomial"
last_step_bs = st.sidebar.checkbox("Black-Scholes sur le dernier pas", value=True)
steps = st.sidebar.slider("Nombre de pas", 10, 300, 100, 10)
prune = st.sidebar.selectbox("Seuil de pruning", [0.0, 1e-7], index=1)

# objets du projet
mkt = Market(spot, vol, rate, div, to_datetime(ex_date))
mdl = Model(to_datetime(pricing_date), steps, prune)
opt = OptionTrade(to_datetime(mat_date), call_put, exercise, strike)
study = Study(mkt, mdl, opt)
has_div = mkt.has_dividend(mdl.price_date, opt.mat_date)

# =============== PRIX ===============
st.header("Prix")
# avec un dividende, study.tree passe tout seul au trinomial
tree = study.tree(steps, trinomial, mkt, prune)
price = tree.price_option(opt, last_step_bs)
price0 = study.tree(steps, trinomial, study.mkt0, prune).price_option(opt, last_step_bs)

if has_div and not trinomial:
    st.info("L'arbre binomial ne gère pas le dividende : le prix est calculé avec l'arbre trinomial.")
if (trinomial or has_div) and tree.negative_proba_weight() > 1e-6:
    st.warning("Des probas de transition sont négatives (dividende grand devant le prix de l'action) :"
               " le prix n'est pas fiable.")

col1, col2, col3 = st.columns(3)
col1.metric("Prix de l'arbre", format(price, ".6f"))
col2.metric("Prix sans dividende", format(price0, ".6f"))
if opt.is_european():
    bs_price = study.bs_price(opt)
    col3.metric("Black-Scholes sans dividende", format(bs_price, ".6f"),
                "écart arbre sans dividende : " + format(price0 - bs_price, ".6f"), delta_color="off")
else:
    col3.metric("Black-Scholes", "pas de formule")

# =============== GRECQUES ===============
st.header("Grecques (arbre trinomial)")
g = Greeks(mkt, mdl, opt, last_step_bs)
# avec Black-Scholes au dernier pas : delta et gamma avec un choc de 1 % sur le spot
# avec la valeur intrinsèque : avec les trois noeuds de la date de pricing (le choc donne un gamma faux)
if last_step_bs:
    delta = g.delta_shift()
    gamma = g.gamma_shift()
else:
    delta = g.delta()
    gamma = g.gamma()
# vega, vomma et vanna pour 1 % de vol, theta pour 1 jour
names = ["Delta", "Gamma", "Vega", "Vomma", "Vanna", "Theta"]
values = [delta, gamma, g.vega() * 0.01, g.vomma() * 0.0001, g.vanna() * 0.01, g.theta()]

# référence Black-Scholes seulement pour une européenne sans dividende
refs = None
if opt.is_european() and not has_div:
    bs = BlackScholes(mkt, mdl.price_date, opt)
    tomorrow = mdl.price_date + datetime.timedelta(days=1)
    theta_bs = BlackScholes(mkt, tomorrow, opt).price() - bs.price()
    refs = [bs.delta(), bs.gamma(), bs.vega() * 0.01, bs.vomma() * 0.0001, bs.vanna() * 0.01, theta_bs]

# tableau : une ligne par grecque, avec tous les chiffres (6 décimales)
table = {"Grecque": names, "Arbre": []}
for v in values:
    table["Arbre"].append(format(v, ".6f"))
if refs is not None:
    table["Black-Scholes"] = []
    for v in refs:
        table["Black-Scholes"].append(format(v, ".6f"))
st.table(table)
st.caption("Vega, vomma et vanna pour 1 % de vol ; theta : prix demain - prix aujourd'hui"
           " (spot moins le dividende si la date ex-dividende est demain).")

# =============== ARBRE ===============
st.header("Arbre")
if steps <= 30:
    st.image(TreeImage(1000, 600, tree, opt).image.as_str())
    st.caption("Blanc : option sans valeur ; du bleu au vert : valeur croissante ;"
               " bord rouge : exercice anticipé.")
else:
    st.write("Choisir 30 pas ou moins pour afficher l'arbre.")

# =============== FRONTIÈRE D'EXERCICE ===============
if opt.is_american():
    st.header("Frontière d'exercice")
    fig = study.exercise_boundary_graph(steps, prune, opt)
    if fig is None:
        st.write("Aucun exercice anticipé : garder l'option vaut toujours au moins autant que l'exercer.")
    else:
        st.pyplot(fig)

# =============== ÉTUDES ===============
# les mêmes graphes que main.py, avec moins de pas pour aller vite
# calculés seulement quand on clique sur le bouton (sinon tout serait recalculé à chaque changement)
st.header("Études")
st.caption("Faites avec un call européen de même strike et même maturité, sur le marché choisi.")
studies = ["1 - Écart x nombre de pas",
           "2 - Écart selon le strike (10 pas)",
           "3 - Américaine - européenne selon le taux",
           "5 - Temps de calcul avec et sans pruning",
           "6 - Dernier pas : écart x nombre de pas",
           "7 - Prix selon le nombre de pas",
           "8 - Probabilité d'être dans la zone d'exercice",
           "11 - Dernier pas : écart selon le strike",
           "12 - Delta selon le nombre de pas",
           "14 - Temps de calcul selon le nombre de pas"]
choice = st.selectbox("Graphe", studies)
if st.button("Afficher le graphe"):
    call = OptionTrade(opt.mat_date, OptionTrade.CALL_LABEL, OptionTrade.EURO_LABEL, strike)
    s = Study(mkt, mdl, call)
    with st.spinner("Calcul en cours..."):
        if choice in (studies[0], studies[5], studies[9]):
            # convergence binomiale et trinomiale de 1 à 100 pas
            s.convergence("binomial", list(range(1, 101)), False)
            s.convergence("trinomial", list(range(1, 101)), True)
        if choice == studies[0]:
            fig = s.convergence_gap_graph()
        elif choice == studies[1]:
            fig = s.strike_gap_graph(10, 0.8 * strike, 1.2 * strike, 0.1)
        elif choice == studies[2]:
            fig = s.rate_graph(100, -0.05, 0.05, 0.01)
        elif choice == studies[3]:
            fig = s.pruning_graph([10, 20, 50, 100, 200, 300, 500], 1e-7)
        elif choice == studies[4]:
            fig = s.last_step_graph([1, 2, 3, 5, 10, 20, 50, 100, 200, 300])
        elif choice == studies[5]:
            fig = s.convergence_price_graph()
        elif choice == studies[6]:
            fig = s.exercise_graph(20)
        elif choice == studies[7]:
            fig = s.last_step_strike_graph(10, 0.7 * strike, 1.3 * strike, 0.25)
        elif choice == studies[8]:
            fig = s.delta_graph(list(range(10, 201, 10)))
        else:
            fig = s.run_time_graph(["binomial", "trinomial"], 50)
    st.pyplot(fig)
