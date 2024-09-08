import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM

# Seuils pour la stratégie
cash_threshold = 0.0145  # Seuil pour entrer en position "cash" dans HMM
leverage = 2  # Levier à appliquer
train_window = 22000  # Taille de la fenêtre d'entraînement (22 000 points de données)

# Actions et leurs pondérations
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Personnalisation des couleurs pour correspondre à la charte graphique
custom_color_palette = ['#D4AF37', '#343a40', '#007bff']

# Télécharger et préparer les données du S&P 500 (^GSPC)
def get_market_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

# Fonction pour télécharger les données des actions
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

# Appliquer la stratégie Long/Short/Cash avec HMM
def apply_long_short_cash_strategy(returns, state_probs, cash_threshold, leverage):
    # Assurons-nous que returns et state_probs ont le même index
    common_index = returns.index.intersection(state_probs.index)
    returns = returns.loc[common_index]
    state_probs = state_probs.loc[common_index]
    
    # state_probs.iloc[:, 0] est la probabilité de l'état haussier
    market_regime = np.where(
        state_probs.iloc[:, 0] > (1 - cash_threshold), 0,  # Long (très probablement haussier)
        np.where(state_probs.iloc[:, 0] < cash_threshold, 1,  # Short (très probablement baissier)
                 2)  # Cash (incertain)
    )
    
    strategy_returns = np.where(
        market_regime == 0, returns * leverage,  # Long
        np.where(market_regime == 1, -returns * leverage,  # Short
                 0)  # Cash
    )
    
    return pd.Series(strategy_returns, index=common_index)

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM.")

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
    state_probs = pd.DataFrame(state_probs, index=test_data.index)

    # Détection du régime actuel et probabilités
    current_state_prob = state_probs.iloc[-1, 0]  # Probabilité de l'état haussier
    current_probs = state_probs.iloc[-1]  # Probabilités des états
    st.subheader('Probabilités de Transition Actuelles')
    st.write(f"Probabilité état haussier (Long) : {current_probs[0]:.2%}")
    st.write(f"Probabilité état baissier (Short) : {current_probs[1]:.2%}")
    
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
    strategy_returns = apply_long_short_cash_strategy(portfolio_returns, state_probs, cash_threshold, leverage)

    # Graphique des probabilités de régimes
    st.subheader("Graphique des Probabilités des Régimes de Marché")
    fig_probs = px.line(state_probs, labels={'index': 'Date', 'value': 'Probabilité'}, 
                        title='Probabilités de Régimes (Haussier vs Baissier)', 
                        color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_probs)

    # Graphique des régimes de marché détectés
    st.subheader("Régimes de Marché Détectés par le HMM")
    test_data['Regime'] = hidden_states
    fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime', 
                             title="Régimes de Marché Détectés", color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_regimes)
