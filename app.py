# streamlit_hmm.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

# Télécharger les données du S&P 500
@st.cache
def load_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    data = data[['Adj Close', 'returns']]
    return data

# Charger les données
st.title('Détection de Régimes de Marché avec un HMM')
st.write("Application basée sur Hidden Markov Model (HMM) pour prédire les régimes de marché haussier ou baissier.")
data = load_data()

# Séparer les données en ensemble d'entraînement et de test
train_window = 24000
train_data = data.iloc[:train_window]
test_data = data.iloc[train_window:]

# Entraîner le modèle HMM
st.write("Entraînement du modèle HMM...")
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100000)
hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))
score_model = hmm_model.score(np.array(train_data['returns']).reshape(-1, 1))
st.write(f"Score du modèle : {score_model}")

# Prédire les régimes de marché
# Assurez-vous que les données sont en 2D avant de passer à hmm_model.predict()
test_data['market_regime'] = hmm_model.predict(np.array(test_data['returns']).reshape(-1, 1))

# Afficher la matrice de transition
transition_matrix = hmm_model.transmat_
st.write("Matrice de transition:")
st.write(transition_matrix)

# Calcul des probabilités de transition
def calculate_next_state_probabilities(hmm_model, current_state_probs):
    transition_matrix = hmm_model.transmat_
    next_state_probs = np.dot(current_state_probs, transition_matrix)
    return next_state_probs

# Calculer les probabilités de l'état actuel
current_state_probs = hmm_model.predict_proba(np.array(test_data['returns'].iloc[-1]).reshape(-1, 1))
st.write("Probabilités de l'état actuel:")
st.write(f"État 0 (Haussier) : {current_state_probs[0][0]:.2%}")
st.write(f"État 1 (Baissier) : {current_state_probs[0][1]:.2%}")

# Calculer les probabilités de transition vers le prochain état
next_state_probs = calculate_next_state_probabilities(hmm_model, current_state_probs[0])
st.write("Probabilités de transition vers le prochain état:")
st.write(f"Vers État 0 (Haussier) : {next_state_probs[0]:.2%}")
st.write(f"Vers État 1 (Baissier) : {next_state_probs[1]:.2%}")

# Visualisation des régimes de marché
st.write("Visualisation des régimes de marché prédits par le HMM :")

palette = {0: 'blue', 1: 'gold'}
plt.figure(figsize=(12, 8))
sns.scatterplot(x=test_data.index, y='Adj Close', hue='market_regime', data=test_data, s=10, palette=palette)
plt.title('Régime de marché')
st.pyplot(plt)

# Afficher les dernières données de régime de marché
st.write("Dernières données de régime de marché:")
st.write(test_data.tail(1))







