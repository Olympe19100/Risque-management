import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analyse du Marché avec HMM", layout="wide")

@st.cache_data
def get_market_data():
    data = yf.download('^GSPC')
    data['returns'] = np.log(data['Adj Close']) - np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)
    return data[['Adj Close', 'returns']]

def calculate_next_state_probabilities(hmm_model, current_state_probs):
    transition_matrix = hmm_model.transmat_
    next_state_probs = np.dot(current_state_probs, transition_matrix)
    return next_state_probs

st.title("Analyse du Marché avec Modèle de Markov Caché")

data = get_market_data()
st.write(f"Nombre total de lignes : {data.shape[0]}")

train_window = 24000
train_data = data.iloc[:train_window]
test_data = data.iloc[train_window:]

np.random.seed(42)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100000)
hmm_model.fit(np.array(train_data['returns']).reshape(-1, 1))

st.write(f"Score Model: {hmm_model.score(np.array(train_data['returns']).reshape(-1, 1)):.2f}")

test_data['market_regime'] = hmm_model.predict(np.array(test_data[['returns']]))

st.subheader("Matrice de Transition")
st.write(pd.DataFrame(hmm_model.transmat_))

current_state_probs = hmm_model.predict_proba(np.array(test_data['returns'].iloc[-1]).reshape(-1, 1))
st.subheader("Probabilités de l'état actuel")
st.write(f"État 0 (Haussier) : {current_state_probs[0][0]:.2%}")
st.write(f"État 1 (Baissier) : {current_state_probs[0][1]:.2%}")

next_state_probs = calculate_next_state_probabilities(hmm_model, current_state_probs[0])
st.subheader("Probabilités de transition vers le prochain état")
st.write(f"Vers État 0 (Haussier) : {next_state_probs[0]:.2%}")
st.write(f"Vers État 1 (Baissier) : {next_state_probs[1]:.2%}")

st.subheader("Visualisation du Régime de Marché")
fig, ax = plt.subplots(figsize=(12, 8))
palette = {0: 'blue', 1: 'Gold'}
sns.scatterplot(x=test_data.index, y='Adj Close', hue='market_regime', data=test_data, s=10, palette=palette, ax=ax)
plt.title('Market regime')
st.pyplot(fig)

st.subheader("Dernière Donnée de Régime de Marché")
st.write(test_data.tail(1))













