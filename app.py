import streamlit as st
import json

# Function to load JSON data from a file
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {file_path}")
    return None

# Load the JSON files
market_regime_data = load_json_file('market_regime.json')
market_regime_analysis_data = load_json_file('market_regime_analysis.json')

# Display market regime data
if market_regime_data:
    st.header("Market Regime Data")
    st.write(f"**Current State:** {market_regime_data['current_state']}")
    st.write(f"**Probability Up:** {market_regime_data['prob_up']:.4f}")
    st.write(f"**Probability Down:** {market_regime_data['prob_down']:.4f}")
    
    # Display mean returns
    st.subheader("Mean Returns")
    for state, return_value in market_regime_data['mean_return'].items():
        st.write(f"State {state}: {return_value:.6f}")
    
    # You might want to display more data from this file, 
    # but since it's quite large, we'll limit it to these key points

# Display market regime analysis data
if market_regime_analysis_data:
    st.header("Market Regime Analysis")
    st.write(f"**Current State:** {market_regime_analysis_data['current_state']}")
    st.write(f"**Market Regime:** {market_regime_analysis_data['market_regime']}")
    st.write(f"**Probability Up:** {market_regime_analysis_data['probability_up']:.4f}")
    st.write(f"**Probability Down:** {market_regime_analysis_data['probability_down']:.4f}")
    st.write(f"**Probability of State Change:** {market_regime_analysis_data['probability_change_state']:.4f}")
    st.write(f"**Regime Duration:** {market_regime_analysis_data['regime_duration']} days")

# If neither file loaded successfully
if not market_regime_data and not market_regime_analysis_data:
    st.error("No data could be loaded. Please check your JSON files.")










