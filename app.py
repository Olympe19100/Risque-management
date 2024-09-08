import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from quantstats.stats import sharpe, max_drawdown

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de Portefeuille avec HMM", layout="wide")

# Définition du portefeuille
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

@st.cache_data
def calculate_portfolio_returns(returns, weights):
    return (returns * weights).sum(axis=1)

def train_hmm_model(returns, n_components=2):
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(returns.values.reshape(-1, 1))
    return model

def main():
    st.title("Analyse de Portefeuille avec Modèle de Markov Caché")

    # Sélection de la période
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Date de début", pd.to_datetime("2010-01-01"))
    end_date = col2.date_input("Date de fin", pd.to_datetime("2023-01-01"))

    if start_date < end_date:
        # Chargement des données
        with st.spinner('Chargement des données...'):
            data, returns = get_stock_data(stocks, start_date, end_date)
            portfolio_returns = calculate_portfolio_returns(returns, pd.Series(stocks))

        # Affichage des rendements du portefeuille
        st.subheader("Rendements du Portefeuille")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio_returns.cumsum())
        ax.set_title("Rendements Cumulatifs du Portefeuille")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rendement Cumulatif")
        st.pyplot(fig)

        # Calcul et affichage des métriques
        sharpe_ratio = sharpe(portfolio_returns)
        max_dd = max_drawdown(portfolio_returns)
        st.write(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        st.write(f"Drawdown Maximum: {max_dd:.2%}")

        # Entraînement et affichage du modèle HMM
        st.subheader("Modèle de Markov Caché")
        try:
            model = train_hmm_model(portfolio_returns)
            hidden_states = model.predict(portfolio_returns.values.reshape(-1, 1))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(portfolio_returns.index, portfolio_returns.cumsum())
            ax.scatter(portfolio_returns.index, portfolio_returns.cumsum(), c=hidden_states, cmap='viridis', alpha=0.5)
            ax.set_title("États Cachés du Modèle HMM sur les Rendements du Portefeuille")
            ax.set_xlabel("Date")
            ax.set_ylabel("Rendement Cumulatif")
            st.pyplot(fig)

            st.write("Informations sur le modèle HMM :")
            st.write(f"Nombre d'états : {model.n_components}")
            st.write(f"Score du modèle : {model.score(portfolio_returns.values.reshape(-1, 1)):.2f}")
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle HMM : {e}")

    else:
        st.error("La date de fin doit être postérieure à la date de début.")

if __name__ == "__main__":
    main()













