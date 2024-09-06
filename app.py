import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from quantstats.stats import sharpe, max_drawdown
from PIL import Image

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

# Calcul des métriques du portefeuille
def calculate_metrics(portfolio_returns):
    sharpe_ratio = sharpe(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized volatility
    return sharpe_ratio, max_dd, volatility

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur des modèles HMM et GMM.")

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

    # Prédiction des régimes de marché avec HMM sur les données de test
    hidden_states_hmm = hmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))
    state_probs_hmm = hmm_model.predict_proba(np.array(test_data['returns']).reshape(-1, 1))
    state_probs_hmm = pd.DataFrame(state_probs_hmm, index=test_data.index)

    # Entraînement du modèle GMM sur les mêmes données
    gmm_model = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))

    # Prédiction des régimes de marché avec GMM sur les données de test
    hidden_states_gmm = gmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))
    state_probs_gmm = gmm_model.predict_proba(np.array(test_data['returns']).reshape(-1, 1))
    state_probs_gmm = pd.DataFrame(state_probs_gmm, index=test_data.index)

    # Télécharger les données des actions
    start_date = test_data.index[0]
    end_date = test_data.index[-1]
    stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

    # Calculer les rendements du portefeuille pondéré
    portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

    # Calcul des métriques du portefeuille
    sharpe_ratio, max_dd, volatility = calculate_metrics(portfolio_returns)
    st.subheader('Métriques du Portefeuille')
    st.write(f"Sharpe Ratio : {sharpe_ratio:.2f}")
    st.write(f"Max Drawdown : {max_dd:.2%}")
    st.write(f"Volatilité (Annualisée) : {volatility:.2%}")

    # Graphique des régimes de marché détectés par HMM
    st.subheader("Régimes de Marché Détectés par le HMM")
    test_data['Regime HMM'] = hidden_states_hmm
    fig_hmm_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime HMM', title="Régimes de Marché Détectés par HMM", color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_hmm_regimes)

    # Graphique des régimes de marché détectés par GMM
    st.subheader("Régimes de Marché Détectés par le GMM")
    test_data['Regime GMM'] = hidden_states_gmm
    fig_gmm_regimes = px.scatter(test_data, x=test_data.index, y='Adj Close', color='Regime GMM', title="Régimes de Marché Détectés par GMM", color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_gmm_regimes)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    fig_pie = px.pie(values=list(stocks.values()), names=list(stocks.keys()), title='Pondérations des Sociétés dans le Portefeuille', color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_pie)

    # Affichage des probabilités de changement de régime pour HMM
    st.subheader("Probabilités de Changement de Régime pour HMM")
    fig_hmm_probs = px.line(state_probs_hmm, title='Probabilités des Régimes de Marché (HMM)',
                            labels={'value': 'Probabilité', 'index': 'Date'},
                            color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_hmm_probs)

    # Affichage des probabilités de changement de régime pour GMM
    st.subheader("Probabilités de Changement de Régime pour GMM")
    fig_gmm_probs = px.line(state_probs_gmm, title='Probabilités des Régimes de Marché (GMM)',
                            labels={'value': 'Probabilité', 'index': 'Date'},
                            color_discrete_sequence=custom_color_palette)
    st.plotly_chart(fig_gmm_probs)

    # Afficher les probabilités du dernier jour pour HMM
    last_day_probs_hmm = state_probs_hmm.iloc[-1]
    st.write("Probabilités de Régime (HMM) pour le Dernier Jour:")
    for regime, prob in enumerate(last_day_probs_hmm):
        st.write(f"Régime {regime}: {prob:.2%}")

    # Afficher les probabilités du dernier jour pour GMM
    last_day_probs_gmm = state_probs_gmm.iloc[-1]
    st.write("Probabilités de Régime (GMM) pour le Dernier Jour:")
    for regime, prob in enumerate(last_day_probs_gmm):
        st.write(f"Régime {regime}: {prob:.2%}")





