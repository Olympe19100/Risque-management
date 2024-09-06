import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
from quantstats.stats import sharpe, max_drawdown
from PIL import Image

# Seuil pour la stratégie
train_window = 22000  # Taille de la fenêtre d'entraînement (22 000 points de données)

# Actions et leurs pondérations
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Charger et afficher le logo
logo = Image.open(r"Olympe Financial group (Logo) (1).png")
st.image(logo, width=200)  # Afficher le logo

# Personnalisation des couleurs pour correspondre à la charte graphique
custom_color_palette = ['#D4AF37', '#343a40', '#007bff']

# Télécharger et préparer les données du S&P 500 (^GSPC) depuis 1951
@st.cache_data
def get_market_data():
    data = yf.download('^GSPC', start='1951-01-01')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

# Fonction pour télécharger les données des actions depuis le 1er janvier 2018
@st.cache_data
def get_stock_data(_tickers, start='2018-01-01', end=None):
    stock_data = {}
    for ticker in _tickers:
        data = yf.download(ticker, start=start, end=end)
        data['Daily Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)
        stock_data[ticker] = data
    return stock_data

# Calcul des rendements pondérés du portefeuille
def calculate_portfolio_returns(stocks, stock_data):
    portfolio_returns = pd.DataFrame(index=stock_data[next(iter(stocks))].index)
    for stock, weight in stocks.items():
        stock_returns = stock_data[stock]['Daily Return']
        portfolio_returns[stock] = stock_returns * (weight / 100)
    portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)
    return portfolio_returns['Portfolio']

# Calcul des métriques pour le portefeuille
def calculate_metrics(returns):
    sharpe_ratio = sharpe(returns)
    max_dd = max_drawdown(returns)
    volatility = returns.std() * np.sqrt(252)  # Annualisée
    return sharpe_ratio, max_dd, volatility

# Fonction modifiée pour appliquer la stratégie Long/Cash
def apply_long_cash_strategy(returns, state_probs):
    # Assurons-nous que returns et state_probs ont le même index
    common_index = returns.index.intersection(state_probs.index)
    returns = returns.loc[common_index]
    state_probs = state_probs.loc[common_index]

    # Appliquer la stratégie : Long sur portefeuille si état 0, cash si état 1
    market_regime = np.where(
        state_probs.iloc[:, 0] > 0.5,  # Si probabilité de l'état haussier (état 0) est supérieure à 0.5
        returns,  # On est long sans levier
        0  # Sinon, on est en cash (rendement 0)
    )
    
    return pd.Series(market_regime, index=common_index)

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM.")

# Télécharger les données du S&P 500 depuis 1951
gspc_data = get_market_data()

# Affichage du nombre total de lignes pour le S&P 500
nombre_lignes = gspc_data.shape[0]
st.write(f"Nombre total de points de données du S&P 500 téléchargés (depuis 1951) : {nombre_lignes}")

# Demander à l'utilisateur d'entrer son montant d'investissement
investment = st.number_input("Montant total de l'investissement (€)", min_value=0.0, value=10000.0)

# Calculer l'allocation sur chaque action en fonction du montant d'investissement
if investment > 0:
    st.subheader('Allocation du portefeuille')
    for stock, weight in stocks.items():
        allocation = (weight / 100) * investment
        st.write(f"{stock} : {allocation:.2f} €")

# Télécharger les données des actions du portefeuille depuis 2018
start_date = '2018-01-01'
end_date = None  # Par défaut, jusqu'à aujourd'hui
stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

# Vérifier que les données ne sont pas vides
if gspc_data.empty or any(data.empty for data in stock_data.values()):
    st.error("Les données sont vides. Veuillez vérifier votre connexion ou la période sélectionnée.")
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
    state_probs = pd.DataFrame(state_probs, index=test_data.index)

    # Détection du régime actuel et probabilités
    current_state_prob = state_probs.iloc[-1, 0]  # Probabilité de l'état haussier
    current_probs = state_probs.iloc[-1]  # Probabilités des états
    st.subheader('Probabilités de Transition Actuelles')
    st.write(f"Probabilité état haussier (Long) : {current_probs[0]:.2%}")
    st.write(f"Probabilité état baissier (Short) : {current_probs[1]:.2%}")
    
    if current_state_prob > 0.5:
        st.info("Régime actuel : Bullish (Haussier). Recommandation : Position Long.")
    else:
        st.warning("Régime actuel : Bearish (Baissier). Recommandation : Position Cash.")

    # Calculer les rendements du portefeuille pondéré à partir de 2018
    portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

    # Appliquer la stratégie Long/Cash
    strategy_returns = apply_long_cash_strategy(portfolio_returns, state_probs)

    # Calcul des métriques du portefeuille sans gestion par CVaR
    sharpe_ratio, max_dd, volatility = calculate_metrics(strategy_returns)
    st.subheader('Métriques du Portefeuille')
    st.write(f"Sharpe Ratio : {sharpe_ratio:.2f}")
    st.write(f"Max Drawdown : {max_dd:.2%}")
    st.write(f"Volatilité (Annualisée) : {volatility:.2%}")

    # Graphique des rendements cumulés avec stratégie Long/Cash
    cumulative_returns = (1 + strategy_returns).cumprod()
    st.subheader('Rendements Cumulés du Portefeuille avec Stratégie Long/Cash')
    fig_returns = px.line(cumulative_returns, title='Rendements Cumulés (Stratégie Long/Cash)', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_returns)

    # Graphique des régimes de marché détectés
    st.subheader("Régimes de Marché Détectés par le HMM")
    test_data['Regime'] = hidden_states
    fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime', title="Régimes de Marché Détectés", color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_regimes)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    fig_pie = px.pie(values=list(stocks.values()), names=list(stocks.keys()), title='Pondérations des Sociétés dans le Portefeuille', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_pie)

