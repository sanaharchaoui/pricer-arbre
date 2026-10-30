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
strike = st.sidebar.number_input("Strike", value=101.0, min_value=0.0, step=1.0)
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
col1.metric("Prix de l'arbre", round(price, 4))
col2.metric("Prix sans dividende", round(price0, 4))
if opt.is_european():
    col3.metric("Black-Scholes sans dividende", round(study.bs_price(opt), 4))
else:
    col3.metric("Black-Scholes", "pas de formule (américaine)")

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

cols = st.columns(6)
for i in range(6):
    if refs is None:
        cols[i].metric(names[i], round(values[i], 5))
    else:
        cols[i].metric(names[i], round(values[i], 5), "BS : " + str(round(refs[i], 5)), delta_color="off")
st.caption("Vega, vomma et vanna pour 1 % de vol ; theta : prix demain - prix aujourd'hui"
           " (spot moins le dividende si la date ex-dividende est demain).")

# =============== ARBRE ===============
st.header("Arbre")
if steps <= 30:
    st.image(TreeImage(1000, 600, tree, opt).image.as_str())
    st.caption("Blanc : option sans valeur ; du bleu au vert : valeur croissante ; bord rouge : exercice anticipé.")
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
