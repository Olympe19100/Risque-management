import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
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
st.image(logo, width=200)

# Personnalisation des couleurs pour correspondre à la charte graphique
custom_color_palette = ['#D4AF37', '#343a40', '#007bff']

# Télécharger et préparer les données du S&P 500 (^GSPC)
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

# Calcul des métriques du portefeuille
def calculate_metrics(portfolio_returns):
    sharpe_ratio = sharpe(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized volatility
    return sharpe_ratio, max_dd, volatility

# Télécharger les données du S&P 500
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM avec des émissions GMM.")

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

    # Entraînement du modèle HMM avec des émissions GMM sur les 22 000 premiers points
    np.random.seed(42)
    n_components = 2  # Nombre de régimes de marché
    n_mix = 3  # Nombre de gaussiennes dans le GMM
    hmm_gmm_model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, covariance_type="full", n_iter=100000)
    hmm_gmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))
    st.write(f"Score du modèle HMM-GMM : {hmm_gmm_model.score(np.array(train_data['returns']).reshape(-1, 1))}")

    # Prédiction des régimes de marché avec HMM-GMM sur les données de test
    hidden_states_gmm = hmm_gmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))
    state_probs_gmm = hmm_gmm_model.predict_proba(np.array(test_data['returns']).reshape(-1, 1))
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

    # Graphique des régimes de marché détectés par HMM-GMM
    st.subheader("Régimes de Marché Détectés par le HMM avec GMM")
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(test_data.index, test_data['Adj Close'], c=hidden_states_gmm, cmap='viridis', alpha=0.6)
    ax.set_title("Régimes de Marché Détectés par HMM-GMM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix Ajusté")
    plt.colorbar(scatter, label='Régime')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(list(stocks.values()), labels=list(stocks.keys()), autopct='%1.1f%%', startangle=90, colors=custom_color_palette)
    ax.set_title('Pondérations des Sociétés dans le Portefeuille')
    st.pyplot(fig)

    # Affichage des probabilités de changement de régime pour HMM-GMM
    st.subheader("Probabilités de Changement de Régime - HMM avec GMM")
    fig, ax = plt.subplots(figsize=(12, 6))
    state_probs_gmm.plot(ax=ax)
    ax.set_title('Probabilités des Régimes de Marché (HMM-GMM)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probabilité')
    plt.legend(title='Régime')
    st.pyplot(fig)

    # Afficher les probabilités du dernier jour pour HMM-GMM
    last_day_gmm_probs = state_probs_gmm.iloc[-1]
    st.write("Probabilités de Régime (HMM-GMM) pour le Dernier Jour:")
    for regime, prob in enumerate(last_day_gmm_probs):
        st.write(f"Régime {regime}: {prob:.2%}")












