import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import quantstats as qs
from scipy.stats import norm

# Portfolio and stock allocation
stocks = {
    'AAPL': 0.76, 'MSFT': 12.85, 'GOOG': 1.68, 'AMZN': 1.74, 'META': 5.26,
    'NVDA': 15.25, 'V': 2.07, 'MA': 3.51, 'BRK-B': 0.53, 'JPM': 1.47,
    'UNH': 28.24, 'BLK': 0.01, 'HD': 2.15, 'T': 0.63, 'PFE': 0.21,
    'MRK': 11.09, 'PEP': 4.47, 'JNJ': 1.72, 'TSLA': 5.83, 'AXP': 0.53
}

# Download S&P 500 data
sp500_data = yf.download('^GSPC')
sp500_data['returns'] = np.log(sp500_data['Adj Close']) - np.log(sp500_data['Adj Close'].shift(1))
sp500_data.dropna(inplace=True)

# Gaussian HMM Model (Pre-trained)
np.random.seed(42)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
hmm_model.fit(np.array(sp500_data['returns']).reshape(-1, 1))
hidden_states = hmm_model.predict(np.array(sp500_data['returns']).reshape(-1, 1))

# Function to calculate CVaR (Conditional Value at Risk)
def calculate_cvar(returns, alpha=0.05):
    var = np.percentile(returns, 100 * alpha)
    cvar = returns[returns <= var].mean()
    return cvar

# Streamlit app layout
st.title("Recommandations de Portefeuille Basées sur Stratégie Long Only")

# Input: Amount to invest
investment = st.number_input("Montant à investir (en $)", min_value=0.0, step=100.0, value=1000.0)

# Display stock allocations
st.subheader("Pondérations des Actions")
stock_df = pd.DataFrame(stocks.items(), columns=['Ticker', 'Allocation (%)'])
for i in range(0, len(stock_df), 5):
    st.write(stock_df.iloc[i:i+5])

# Calculate amounts to invest per stock
stock_df['Investment'] = stock_df['Allocation (%)'] / 100 * investment
st.subheader("Montants à Investir par Action")
st.write(stock_df[['Ticker', 'Investment']])

# Plot market regime (Long vs Cash)
st.subheader("Visualisation des Régimes de Marché")
fig, ax = plt.subplots(figsize=(10, 6))
sp500_data['Adj Close'].plot(ax=ax, label='S&P 500')
colors = ['gray' if state == 0 else 'blue' for state in hidden_states]
ax.scatter(sp500_data.index, sp500_data['Adj Close'], color=colors, s=10, label='Regime')
ax.set_title("Régimes de Marché (Long vs Cash)")
st.pyplot(fig)

# Calculate and display CVaR
cvar_tolerable = -0.02  # Example: set a tolerable CVaR
cvar_current = calculate_cvar(sp500_data['returns'])
st.write(f"CVaR actuel: {cvar_current:.2%}")
st.write(f"CVaR tolérable: {cvar_tolerable:.2%}")

# Display market indices
st.subheader("Indices Boursiers Internationaux")
indices = ['^GSPC', '^GDAXI', '^FTSE', '^N225']  # S&P 500, DAX, FTSE, Nikkei
index_data = {index: yf.Ticker(index).history(period='1d')['Close'][0] for index in indices}
index_df = pd.DataFrame(index_data.items(), columns=['Indice', 'Prix Actuel'])
st.write(index_df)

# Fetch and display news for stocks
st.subheader("Dernières Nouvelles pour les Actions")
for ticker in stock_df['Ticker']:
    stock_info = yf.Ticker(ticker)
    news = stock_info.news
    st.write(f"**{ticker}**:")
    if news:
        for article in news[:2]:  # Display the top 2 news articles
            st.write(f"- [{article['title']}]({article['link']})")
    else:
        st.write("Aucune nouvelle récente.")

st.write("Analyse terminée. Les graphiques ont été affichés et les recommandations fournies.")
