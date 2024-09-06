import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
from quantstats.stats import sharpe, max_drawdown
from PIL import Image

# Seuils pour la stratégie
cash_threshold = 0.0208  # Mise à jour du seuil pour l'état "cash"
cvar_threshold = 0.0569
leverage = 2
train_window = 22000

# Actions et leurs pondérations (inchangées)
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Charger et afficher le logo
logo = Image.open(r"C:\Users\Hamid\Desktop\Trading Olympe\Olympe Financial group (Logo) (1).png")
st.image(logo, width=200)

custom_color_palette = ['#D4AF37', '#343a40', '#007bff']

@st.cache_data
def get_market_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

# Autres fonctions inchangées...

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM et gestion des risques via la CVaR.")

gspc_data = get_market_data()

# Affichage du nombre total de lignes
nombre_lignes = gspc_data.shape[0]
st.write(f"Nombre total de points de données du S&P 500 téléchargés : {nombre_lignes}")

# Demander à l'utilisateur d'entrer son montant d'investissement
investment = st.number_input("Montant total de l'investissement (€)", min_value=0.0, value=10000.0)

# Calculer l'allocation sur chaque action en fonction du montant d'investissement
if investment > 0:
    st.subheader('Allocation du portefeuille')
    for stock, weight in stocks.items():
        allocation = (weight / 100) * investment
        st.write(f"{stock} : {allocation:.2f} €")

# Vérifier que les données ne sont pas vides
if gspc_data.empty:
    st.error("Les données du S&P 500 sont vides.")
else:
    # Diviser les données en entraînement (22 000 points) et test
    train_data = gspc_data.iloc[:train_window]
    test_data = gspc_data.iloc[train_window:]

    # Entraînement du modèle HMM sur les 22 000 premiers points
    np.random.seed(42)
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100000)
    hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))
    st.write(f"Score du modèle HMM : {hmm_model.score(np.array(train_data['returns']).reshape(-1, 1))}")

    # Prédiction des régimes de marché sur les données de test
    hidden_states = hmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))
    state_probs = hmm_model.predict_proba(np.array(test_data['returns']).reshape(-1, 1))
    
    # Créer une nouvelle colonne pour le régime de marché incluant l'état "cash"
    test_data['market_regime'] = np.where(
        (state_probs[:, 0] > cash_threshold) & (state_probs[:, 0] < (1 - cash_threshold)),
        2,  # État "cash"
        hidden_states
    )

    # Affichage de la matrice de transition
    transition_matrix = hmm_model.transmat_
    st.subheader('Matrice de Transition')
    st.write(pd.DataFrame(transition_matrix))

    # Détection du régime actuel et probabilités
    current_state_prob = state_probs[-1, 0]  # Probabilité de l'état haussier
    st.subheader('Probabilités de Transition Actuelles')
    st.write(f"Probabilité état haussier (Long) : {current_state_prob:.2%}")
    st.write(f"Probabilité état baissier (Short) : {1 - current_state_prob:.2%}")
    
    if current_state_prob > (1 - cash_threshold):
        st.info("Régime actuel : Bullish (Haussier). Recommandation : Position Long.")
    elif current_state_prob < cash_threshold:
        st.warning("Régime actuel : Bearish (Baissier). Recommandation : Position Short.")
    else:
        st.info("Régime actuel : Incertain. Recommandation : Position Cash.")

    # Télécharger les données des actions
    start_date = test_data.index[0]
    end_date = test_data.index[-1]
    stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

    # Calculer les rendements du portefeuille pondéré
    portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

    # Appliquer la stratégie Long/Short/Cash
    strategy_returns = apply_long_short_cash_strategy(portfolio_returns, pd.DataFrame(state_probs, index=test_data.index), cash_threshold, leverage)

    # Application de la gestion des risques basée sur la CVaR
    managed_returns = apply_cvar_risk_management(strategy_returns, cvar_threshold)

    # Calcul des métriques du portefeuille géré
    sharpe_ratio, max_drawdown, volatility = calculate_metrics(managed_returns)
    st.subheader('Métriques du Portefeuille Géré')
    st.write(f"Sharpe Ratio : {sharpe_ratio:.2f}")
    st.write(f"Max Drawdown : {max_drawdown:.2%}")
    st.write(f"Volatilité (Annualisée) : {volatility:.2%}")

    # Calcul de la CVaR sur les rendements du portefeuille géré
    cvar = calculate_cvar(managed_returns)
    st.subheader('Analyse de la CVaR du Portefeuille Géré')
    st.write(f"CVaR actuel : {cvar:.2%}")
    st.write(f"Seuil de CVaR : {cvar_threshold:.2%}")

    # Recommandation basée sur la CVaR
    if cvar > cvar_threshold:
        st.error(f"CVaR dépasse le seuil de {cvar_threshold:.2%}. Recommandation : Surveiller de près les positions.")
    else:
        st.success(f"CVaR sous contrôle ({cvar:.2%}).")

    # Graphique des rendements gérés avec stratégie Long/Short/Cash et CVaR
    cumulative_managed_returns = (1 + managed_returns).cumprod()
    st.subheader('Rendements Cumulés du Portefeuille avec Stratégie Long/Short/Cash et Gestion des Risques')
    fig_cvar = px.line(cumulative_managed_returns, title='Rendements Cumulés (Stratégie Long/Short/Cash avec Gestion des Risques)', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_cvar)

    # Graphique des régimes de marché détectés
    st.subheader("Régimes de Marché Détectés par le HMM")
    fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='market_regime', 
                             title="Régimes de Marché Détectés", 
                             color_discrete_map={0: 'blue', 1: 'Gold', 2: 'gray'},
                             labels={'market_regime': 'Régime de Marché'})
    st.plotly_chart(fig_regimes)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    fig_pie = px.pie(values=list(stocks.values()), names=list(stocks.keys()), title='Pondérations des Sociétés dans le Portefeuille', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_pie)

