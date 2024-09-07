import streamlit as st
import pandas as pd
import json
import requests

# Title for the app
st.title("Market Regime Analysis and News Sentiment Dashboard")

# Load the Market Regime Analysis file from GitHub
def load_market_regime():
    url = "https://raw.githubusercontent.com/votre-utilisateur/market-analysis/main/market_regime_analysis.json"
    response = requests.get(url)
    return response.json()

market_regime_data = load_market_regime()

# Display market regime details
st.write(f"**Current Market Regime:** {market_regime_data['market_regime']}")
st.write(f"**Probability of Market Going Up:** {market_regime_data['probability_up']:.2%}")
st.write(f"**Probability of Market Going Down:** {market_regime_data['probability_down']:.2%}")
st.write(f"**Probability of Regime Change:** {market_regime_data['probability_change_state']:.2%}")
st.write(f"**Current Regime Duration:** {market_regime_data['regime_duration']} days")

# Plot the probabilities
probabilities = {
    'Up': market_regime_data['probability_up'],
    'Down': market_regime_data['probability_down'],
    'Change State': market_regime_data['probability_change_state']
}
st.bar_chart(list(probabilities.values()), labels=list(probabilities.keys()))

# Load and display top articles
st.subheader("Top Articles by Sentiment")

def load_top_articles():
    url = "https://raw.githubusercontent.com/votre-utilisateur/market-analysis/main/top_articles.json"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

top_articles_df = load_top_articles()

# Convert the date from timestamp to a readable format
top_articles_df['Date'] = pd.to_datetime(top_articles_df['Date'], unit='ms')

# Show the top 5 articles
st.write(top_articles_df[['Date', 'Title', 'Source', 'Sentiment', 'Adjusted_Score']].head())

# Sentiment analysis chart
st.subheader("Sentiment Distribution")
sentiment_counts = top_articles_df['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)









