import pandas as pd
import numpy as np

# Load the CSV data and preprocess it
def load_data():
    # Read the CSV file
    data = pd.read_csv("s&p500_data.csv")
    
    # Replace "#DIV/0!" with 0 in the 'Change Rounded' column
    data['Change Rounded'] = data['Change Rounded'].apply(lambda x: 0 if x == "#DIV/0!" else x)
    
    # Convert 'Change Rounded' to numeric, coercing errors into NaN and then replacing them with 0
    data['Change Rounded'] = pd.to_numeric(data['Change Rounded'], errors='coerce').fillna(0)
    
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)

    # Extract relevant columns for Markov Chain
    df = data[['Date', 'Change Rounded']]
    df.rename(columns={'Change Rounded': 'State'}, inplace=True)
    
    # Create a state map
    states = df['State'].unique()
    state_map = {state: idx for idx, state in enumerate(states)}
    df['StateID'] = df['State'].map(state_map)

    return df

# Build transition matrices for higher-order Markov chains
def build_transition_matrix(df, k_order):
    transitions = {i: np.zeros((len(df['State'].unique()),) * (i+1)) for i in range(1, k_order+1)}

    for order in range(1, k_order+1):
        for i in range(len(df) - order):
            current_states = tuple(df.iloc[i:i+order]['StateID'])
            next_state = df.iloc[i+order]['StateID']
            transitions[order][current_states + (next_state,)] += 1

    # Normalize transition matrices to probabilities
    for order in transitions:
        sums = transitions[order].sum(axis=-1, keepdims=True)
        transitions[order] = np.divide(transitions[order], sums, where=sums != 0)

    return transitions

# Function to predict the next state (percentage change)
def predict_next_state(current_states, transitions, k_order):
    probs = transitions[k_order][tuple(current_states)]
    predicted_state = np.argmax(probs)
    predicted_change = states[predicted_state]  # Reverse mapping to the actual value
    prob = probs[predicted_state]  # The probability of the prediction
    return predicted_change, prob
