import streamlit as st
import requests
import json

# Fonction pour charger les données JSON depuis une URL avec gestion d'erreurs
def load_json_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifier si la requête a échoué
        return response.json()  # Tenter de décoder les données JSON
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return None
    except json.decoder.JSONDecodeError:
        st.error("Erreur de décodage JSON. Veuillez vérifier le format des données.")
        return None

# URL du fichier JSON sur GitHub
url_market_regime = "https://raw.githubusercontent.com/votre-utilisateur/votre-repo/main/market_regime_analysis.json"

# Charger les données du marché
market_regime_data = load_json_from_github(url_market_regime)

if market_regime_data:
    # Afficher les informations si les données ont été chargées correctement
    st.write(f"**Current Market Regime:** {market_regime_data['market_regime']}")
    st.write(f"**Probability of Market Going Up:** {market_regime_data['probability_up']:.2%}")
    st.write(f"**Probability of Market Going Down:** {market_regime_data['probability_down']:.2%}")
    st.write(f"**Probability of Regime Change:** {market_regime_data['probability_change_state']:.2%}")
    st.write(f"**Current Regime Duration:** {market_regime_data['regime_duration']} days")
else:
    st.error("Les données du marché n'ont pas pu être chargées.")










