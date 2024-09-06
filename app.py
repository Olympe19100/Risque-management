import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
from PIL import Image

# Seuils pour la stratégie simplifiée
train_window = 22000  # Taille de la fenêtre d'entraînement (22 000 points de données)
leverage = 1  # Pas de levier pour la stratégie buy and hold

# Actions et leurs pondérations
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Télécharger et préparer les données du S&P 500 (^GSPC)
@st.cache_data
def get_market_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

# Télécharger les données des actions
@st.cache_data
def get_stock_data(_tickers, start, end):
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

# Stratégie simplifiée Buy and Hold et Cash
def apply_buy_and_hold_cash_strategy(returns, hidden_states):
    strategy_returns = np.where(hidden_states == 1, 0, returns)  # En cash si état baissier (1)
    return pd.Series(strategy_returns, index=returns.index)

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Stratégie Buy and Hold et Cash")
st.write("Analyse des rendements du portefeuille avec une stratégie Buy and Hold et Cash en régime baissier.")

gspc_data = get_market_data()

# Diviser les données en entraînement (22 000 points) et test
train_data = gspc_data.iloc[:train_window]
test_data = gspc_data.iloc[train_window:]

# Entraînement du modèle HMM sur les 22 000 premiers points
np.random.seed(42)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100000)
hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))

# Prédiction des régimes de marché sur les données de test
hidden_states = hmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))

# Télécharger les données des actions pour la période de test
start_date = test_data.index[0]
end_date = test_data.index[-1]
stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

# Calculer les rendements du portefeuille pondéré
portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

# Appliquer la stratégie Buy and Hold et Cash
strategy_returns = apply_buy_and_hold_cash_strategy(portfolio_returns, hidden_states)

# Calcul des rendements cumulés
cumulative_returns = (1 + strategy_returns).cumprod()

# Graphique des rendements cumulés
st.subheader('Rendements Cumulés avec Stratégie Buy and Hold et Cash')
fig = px.line(cumulative_returns, title='Rendements Cumulés (Stratégie Buy and Hold et Cash)', color_discrete_sequence=['#D4AF37'])
st.plotly_chart(fig)

# Graphique des régimes de marché détectés
st.subheader("Régimes de Marché Détectés par le HMM")
test_data['Regime'] = hidden_states
fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime', title="Régimes de Marché Détectés", color_discrete_sequence=['#D4AF37', '#343a40'])
st.plotly_chart(fig_regimes)





