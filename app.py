import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
from PIL import Image
from sklearn.linear_model import LinearRegression

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

# Télécharger et préparer les données du S&P 500 (^GSPC)
@st.cache_data
def get_market_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

# Fonction pour télécharger les données des actions
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

# Fonction pour calculer le Beta de chaque action par rapport au S&P 500
def calculate_betas(stock_data, gspc_data):
    betas = {}
    gspc_returns = gspc_data['returns'].iloc[1:].values.reshape(-1, 1)  # Retours du S&P 500
    for stock in stock_data:
        stock_returns = stock_data[stock]['Daily Return'].values.reshape(-1, 1)[1:]
        reg = LinearRegression().fit(gspc_returns, stock_returns)
        betas[stock] = reg.coef_[0][0]
    return betas

# Calcul du Beta du portefeuille
def calculate_portfolio_beta(betas, stocks):
    portfolio_beta = 0
    for stock, weight in stocks.items():
        portfolio_beta += betas[stock] * (weight / 100)
    return portfolio_beta

# Stress Testing basé sur le Beta du portefeuille
def stress_test_with_beta(portfolio_beta, stress_level):
    return portfolio_beta * stress_level

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM avec Stress Test et Beta du portefeuille.")

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
    train_window = 22000  # Taille de la fenêtre d'entraînement
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

    # Télécharger les données des actions
    start_date = test_data.index[0]
    end_date = test_data.index[-1]
    stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

    # Calculer les rendements du portefeuille pondéré
    portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

    # Calcul des betas des actions par rapport au S&P 500
    betas = calculate_betas(stock_data, gspc_data)
    st.write("Betas des actions par rapport au S&P 500 :")
    for stock, beta in betas.items():
        st.write(f"{stock} : {beta:.2f}")

    # Calcul du beta du portefeuille
    portfolio_beta = calculate_portfolio_beta(betas, stocks)
    st.write(f"Beta du portefeuille : {portfolio_beta:.2f}")

    # Stress Testing avec plusieurs scénarios (chute de 10%, 20%, 30%)
    st.subheader('Stress Testing : Scénarios de Chute du Marché')
    for stress_level in [-0.1, -0.2, -0.3]:  # Chute de 10%, 20%, 30%
        stressed_return = stress_test_with_beta(portfolio_beta, stress_level)
        st.write(f"Scénario de Chute de {int(abs(stress_level * 100))}% du S&P 500 :")
        st.write(f"Rendement simulé du portefeuille : {stressed_return:.2%}")

    # Graphique des régimes de marché détectés
    st.subheader("Régimes de Marché Détectés par le HMM")
    test_data['Regime'] = hidden_states
    fig_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime', title="Régimes de Marché Détectés", color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_regimes)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    fig_pie = px.pie(values=list(stocks.values()), names=list(stocks.keys()), title='Pondérations des Sociétés dans le Portefeuille', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_pie)

    # Affichage des probabilités de changement de régime
    st.subheader("Probabilités de Changement de Régime")

    # Graphique des probabilités des régimes de marché
    fig_probs = px.line(state_probs, title='Probabilités des Régimes de Marché',
                        labels={'value': 'Probabilité', 'index': 'Date'},
                        color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_probs)

    # Afficher les probabilités du dernier jour
    last_day_probs = state_probs.iloc[-1]
    st.write("Probabilités de Régime pour le Dernier Jour:")
    for regime, prob in enumerate(last_day_probs):
        st.write(f"Régime {regime}: {prob:.2%}")

