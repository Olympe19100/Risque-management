import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
import matplotlib.pyplot as plt

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




