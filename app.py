import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
import plotly.express as px

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
    data = yf.download('^GSPC', start='1951-01-01')
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

# Télécharger les données du S&P 500
st.title("Analyse des Régimes de Marché avec HMM (2 états)")
gspc_data = get_market_data()

# Vérifier que les données ne sont pas vides
if gspc_data.empty:
    st.error("Les données du S&P 500 sont vides.")
else:
    # Diviser les données en entraînement et test
    train_window = int(len(gspc_data) * 0.8)  # Utiliser 80% des données pour l'entraînement
    train_data = gspc_data.iloc[:train_window]
    test_data = gspc_data.iloc[train_window:]

    # Entraînement du modèle HMM avec des émissions gaussiennes
    np.random.seed(42)
    n_components = 2  # Nombre de régimes de marché (modifié à 2)
    hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
    hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))
    
    st.write(f"Score du modèle HMM : {hmm_model.score(np.array(train_data['returns']).reshape(-1, 1))}")

    # Prédiction des régimes de marché avec HMM sur les données de test
    hidden_states = hmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))
    state_probs = hmm_model.predict_proba(np.array(test_data['returns']).reshape(-1, 1))
    state_probs = pd.DataFrame(state_probs, index=test_data.index)

    # Afficher les régimes détectés
    test_data['Regime'] = hidden_states
    fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime',
                             title="Régimes de Marché Détectés par HMM (2 états)")
    st.plotly_chart(fig_regimes)

    # Affichage des probabilités de changement de régime
    st.subheader("Probabilités de Changement de Régime - HMM (2 états)")
    fig_probs = px.line(state_probs, title='Probabilités des Régimes de Marché (HMM - 2 états)',
                        labels={'value': 'Probabilité', 'index': 'Date'})
    st.plotly_chart(fig_probs)

    # Afficher les probabilités du dernier jour
    last_day_probs = state_probs.iloc[-1]
    st.write("Probabilités de Régime pour le Dernier Jour :")
    for regime, prob in enumerate(last_day_probs):
        st.write(f"Régime {regime}: {prob:.2%}")

    # Utiliser ces probabilités pour de la gestion de portefeuille ou des alertes
    if last_day_probs.idxmax() == 0:
        st.warning("Le marché semble être dans un régime baissier.")
    else:
        st.success("Le marché est dans un régime haussier.")



