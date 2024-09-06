import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from quantstats.stats import sharpe, max_drawdown
from PIL import Image

# Seuils pour la stratégie
cash_threshold = 0.0156
cvar_threshold = 0.0238
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

# Calcul des métriques pour le portefeuille
def calculate_metrics(returns):
    sharpe_ratio = sharpe(returns)
    max_dd = max_drawdown(returns)
    volatility = returns.std() * np.sqrt(252)  # Annualisée
    return sharpe_ratio, max_dd, volatility

# Fonction pour calculer la CVaR
def calculate_cvar(returns, confidence_level=0.95, window=252):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = -sorted_returns[:index].mean()  # Moyenne des pires pertes
    return cvar

# Fonction pour appliquer la gestion des risques basée sur CVaR
def apply_cvar_risk_management(returns, cvar_threshold, window=252):
    cvar_series = returns.rolling(window=window).apply(lambda x: calculate_cvar(x, window=window), raw=False)
    risk_management_exit = cvar_series > cvar_threshold  # Si CVaR dépasse le seuil, on sort du marché
    managed_returns = returns.copy()
    managed_returns[risk_management_exit] = 0  # Appliquer la gestion en remplaçant les rendements par 0
    return managed_returns

# Nouvelle fonction pour appliquer la stratégie buy and hold
def apply_buy_hold_strategy(returns, state_probs, cash_threshold):
    common_index = returns.index.intersection(state_probs.index)
    returns = returns.loc[common_index]
    state_probs = state_probs.loc[common_index]
    
    market_regime = np.where(state_probs.iloc[:, 0] > (1 - cash_threshold), 1, 0)  # 1 pour buy, 0 pour cash
    
    strategy_returns = np.where(market_regime == 1, returns, 0)  # Appliquer les rendements seulement en position buy
    
    return pd.Series(strategy_returns, index=common_index)

# Interface utilisateur Streamlit
st.title("Olympe Financial Group - Tableau de Bord")
st.write("Analyse des rendements du portefeuille basé sur un modèle HMM et gestion des risques via la CVaR.")

# Télécharger les données du S&P 500
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

    # Télécharger les données des actions
    start_date = test_data.index[0]
    end_date = test_data.index[-1]
    stock_data = get_stock_data(list(stocks.keys()), start=start_date, end=end_date)

    # Calculer les rendements du portefeuille pondéré
    portfolio_returns = calculate_portfolio_returns(stocks, stock_data)

    # Appliquer la stratégie Buy and Hold
    strategy_returns = apply_buy_hold_strategy(portfolio_returns, state_probs, cash_threshold)

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

    # Graphique des rendements gérés avec stratégie Buy and Hold et CVaR
    cumulative_managed_returns = (1 + managed_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_managed_returns.index, cumulative_managed_returns.values)
    ax.set_title('Rendements Cumulés (Stratégie Buy and Hold avec Gestion des Risques)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rendements Cumulés')
    st.pyplot(fig)

    # Graphique des régimes de marché détectés avec Matplotlib
    st.subheader("Régimes de Marché Détectés par le HMM")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(test_data.index, test_data['Adj Close'], c=hidden_states, cmap='viridis')
    ax.set_title("Régimes de Marché Détectés")
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix Ajusté de Clôture')
    st.pyplot(fig)

    # Graphique en camembert des pondérations du portefeuille
    st.subheader('Pondérations du Portefeuille')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(list(stocks.values()), labels=list(stocks.keys()), autopct='%1.1f%%', startangle=90)
    ax.set_title('Pondérations des Sociétés dans le Portefeuille')
    st.pyplot(fig)

    # Détection du régime actuel et probabilités
    current_state_prob = state_probs.iloc[-1, 0]  # Probabilité de l'état haussier
    current_probs = state_probs.iloc[-1]  # Probabilités des états
    st.subheader('Probabilités de Transition Actuelles')
    st.write(f"Probabilité état haussier (Buy) : {current_probs[0]:.2%}")
    st.write(f"Probabilité état baissier (Cash) : {current_probs[1]:.2%}")

    if current_state_prob > (1 - cash_threshold):
        st.info("Régime actuel : Bullish (Haussier). Recommandation : Conserver les positions (Buy).")
    else:
        st.warning("Régime actuel : Bearish (Baissier) ou Incertain. Recommandation : Détenir en espèces (Cash).")

