import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
import matplotlib.pyplot as plt
import vectorbt as vbt
import quantstats as qs
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Téléchargement des données S&P 500
st.title("Analyse du marché avec HMM et Portfolio Management")

# Télécharger les données du S&P 500
data = yf.download('^GSPC')
data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
data.dropna(inplace=True)
data = data[['Adj Close', 'returns']]

st.write("Aperçu des données du S&P 500 :")
st.write(data.head())

# Diviser les données en ensemble d'entraînement et de test
train_window = 24000
train_data = data.iloc[:train_window]
test_data = data.iloc[train_window:]

# Appliquer le modèle de Markov caché (HMM)
np.random.seed(42)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100000)
hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))

st.write(f"Score du modèle HMM : {hmm_model.score(np.array(train_data['returns']).reshape(-1, 1)):.2f}")

# Prédiction des régimes de marché pour les données de test
test_data['market_regime'] = hmm_model.predict(np.array(test_data[['returns']]))
st.write("Dernières données de régime de marché :")
st.write(test_data.tail())

# Affichage de la matrice de transition
transition_matrix = hmm_model.transmat_
st.write("Matrice de transition (probabilités de transition entre les états) :")
st.write(pd.DataFrame(transition_matrix, columns=[f"État {i}" for i in range(transition_matrix.shape[0])], index=[f"État {i}" for i in range(transition_matrix.shape[0])]))

# Visualisation des régimes de marché
st.subheader("Visualisation des régimes de marché")

plt.figure(figsize=(12, 8))
palette = {0: 'blue', 1: 'gold'}
sns.scatterplot(x=test_data.index, y='Adj Close', hue='market_regime', data=test_data, s=10, palette=palette)
plt.title('Régimes de marché')
st.pyplot(plt)

# Définir les entrées et sorties de marché pour le portefeuille
prices = test_data['Adj Close']
hmm_entries_long = (test_data['market_regime'] == 0)
hmm_exits_long = (test_data['market_regime'] == 1)

hmm_portfolio = vbt.Portfolio.from_signals(prices, entries=hmm_entries_long, exits=hmm_exits_long)
hmm_returns = hmm_portfolio.returns()

# Comparer aux rendements du benchmark
benchmark_returns = prices.pct_change().dropna()

# Performance du portefeuille
st.subheader("Performance du portefeuille avec HMM")
st.write(f"Performance cumulée du portefeuille sur les 5 dernières années : {hmm_returns.cumsum()[-1]:.2%}")

# Distribution des rendements
extreme_market = test_data[test_data['market_regime'] == 1]
normal_market = test_data[test_data['market_regime'] == 0]

plt.figure(figsize=(12, 6))
sns.histplot(extreme_market['returns'], color='red', bins=30, kde=True, element='step', label='Marchés Extrêmes')
sns.histplot(normal_market['returns'], color='blue', bins=30, kde=True, element='step', label='Marchés Normaux')
plt.axvline(extreme_market['returns'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(normal_market['returns'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.title('Distribution des Rendements')
plt.xlabel('Rendements')
plt.ylabel('Fréquence')
plt.legend()
st.pyplot(plt)

# Analyse de la performance du portefeuille avec Quantstats
qs.extend_pandas()

st.subheader("Statistiques de performance")

sharpe_ratio = hmm_returns.sharpe()
sortino_ratio = hmm_returns.sortino()
max_drawdown = hmm_returns.max_drawdown()

st.write(f"Sharpe Ratio : {sharpe_ratio:.2f}")
st.write(f"Sortino Ratio : {sortino_ratio:.2f}")
st.write(f"Max Drawdown : {max_drawdown:.2%}")

# Générer un rapport complet
st.subheader("Rapport complet de performance")
report_html = 'rapport_performance.html'
qs.reports.html(hmm_returns, benchmark=benchmark_returns, output=report_html)

with open(report_html, 'rb') as f:
    st.download_button("Télécharger le rapport de performance", data=f, file_name=report_html)

