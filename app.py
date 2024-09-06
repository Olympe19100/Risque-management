import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# Titre de l'application
st.title("Gestion de Portefeuille Basée sur les Régimes de Marché (HMM)")

# Télécharger les données du S&P 500 (SPY)
start_date = "2019-01-24"
end_date = "2024-04-16"
st.write("Téléchargement des données du S&P 500 (SPY)...")
spy_data = yf.download('SPY', start=start_date, end=end_date)

# Calculer les rendements journaliers du S&P 500
spy_data['Daily Return'] = spy_data['Adj Close'].pct_change().dropna()

# Entraîner le modèle HMM pour prédire les régimes de marché
st.write("Entraînement du modèle HMM sur les rendements du SPY...")
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
hmm_model.fit(spy_data['Daily Return'].dropna().values.reshape(-1, 1))

# Prédire les régimes de marché
hidden_states = hmm_model.predict(spy_data['Daily Return'].dropna().values.reshape(-1, 1))
spy_data = spy_data.dropna()  # Suppression des valeurs NaN pour correspondre aux états cachés
spy_data['Market Regime'] = hidden_states

# Afficher les régimes de marché détectés
st.subheader("Régimes de Marché Détectés par HMM")
st.write(spy_data[['Adj Close', 'Market Regime']].tail())

# Définir les symboles des actions dans le portefeuille et leurs allocations
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Créer un DataFrame pour stocker les rendements pondérés quotidiens
st.write("Calcul des rendements pondérés pour le portefeuille...")
portfolio_returns = pd.DataFrame()

for stock, allocation in stocks.items():
    data = yf.download(stock, start=start_date, end=end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    portfolio_returns[stock] = data['Daily Return'] * (allocation / 100)

# Calculer le rendement quotidien du portefeuille
portfolio_returns['Portfolio Daily Return'] = portfolio_returns.sum(axis=1)

# Calculer le rendement cumulatif du portefeuille
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Portfolio Daily Return']).cumprod()

# Assurer que la taille du DataFrame corresponde aux données du SPY
portfolio_returns = portfolio_returns.loc[spy_data.index]

# Définir les signaux d'entrée et de sortie en fonction des régimes de marché
st.write("Application de la stratégie basée sur les régimes de marché...")
entries = spy_data['Market Regime'] == 0  # Entrer dans le marché quand le régime est haussier
exits = spy_data['Market Regime'] == 1   # Sortir du marché quand le régime est baissier

# Appliquer ces signaux pour calculer le rendement actif
portfolio_returns['Active Return'] = portfolio_returns['Portfolio Daily Return'].where(entries, 0)

# Calculer le rendement cumulatif actif
portfolio_returns['Active Cumulative Return'] = (1 + portfolio_returns['Active Return']).cumprod()

# Afficher la performance cumulée
cumulative_performance = portfolio_returns['Active Cumulative Return'].iloc[-1]
st.subheader(f"La performance cumulée du portefeuille avec gestion des régimes sur 5 ans est de {cumulative_performance:.2%}")

# Tracer les rendements cumulés du portefeuille avec et sans la stratégie de régimes
st.subheader("Graphique des Rendements Cumulés")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(portfolio_returns['Cumulative Return'], label='Rendement Cumulé (Sans Stratégie)')
ax.plot(portfolio_returns['Active Cumulative Return'], label='Rendement Cumulé (Avec Régimes)', linestyle='--')
ax.set_title('Rendement Cumulé du Portefeuille')
ax.legend()
st.pyplot(fig)

# Afficher les régimes de marché dans un graphique
st.subheader("Graphique des Régimes de Marché Détectés par HMM")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(spy_data['Adj Close'], label='Prix du SPY')
ax.scatter(spy_data.index, spy_data['Adj Close'], c=spy_data['Market Regime'], cmap='coolwarm', label='Régimes de Marché')
ax.set_title("Régimes de Marché Détectés")
ax.legend()
st.pyplot(fig)




