import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def load_data():
    """Load and preprocess the S&P 500 data"""
    try:
        # Read the CSV file
        data = pd.read_csv("s&p500_data.csv")
        
        # Convert Date column to datetime and sort by date
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Clean the Change Rounded column
        data['Change Rounded'] = data['Change Rounded'].replace('#DIV/0!', 0)
        data['Change Rounded'] = pd.to_numeric(data['Change Rounded'], errors='coerce').fillna(0)
        
        print(f"Loaded {len(data)} data points from {data['Date'].min()} to {data['Date'].max()}")
        print(f"Change range: {data['Change Rounded'].min():.3f} to {data['Change Rounded'].max():.3f}")
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError("s&p500_data.csv file not found")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def create_states(data, bin_size=0.25):
    """Convert continuous percentage changes into discrete states"""
    changes = data['Change Rounded'].values
    
    # Create bins for the states
    min_change = changes.min()
    max_change = changes.max()
    
    # Create state labels based on binned ranges
    def get_state(change):
        if change <= -2.0:
            return "Large Drop (-2%+)"
        elif change <= -1.0:
            return "Drop (-1% to -2%)"
        elif change <= -0.5:
            return "Small Drop (-0.5% to -1%)"
        elif change <= 0.0:
            return "Slight Drop (0% to -0.5%)"
        elif change <= 0.5:
            return "Slight Rise (0% to 0.5%)"
        elif change <= 1.0:
            return "Small Rise (0.5% to 1%)"
        elif change <= 2.0:
            return "Rise (1% to 2%)"
        else:
            return "Large Rise (2%+)"
    
    states = [get_state(change) for change in changes]
    unique_states = list(set(states))
    
    print(f"Created {len(unique_states)} unique states")
    for state in unique_states:
        count = states.count(state)
        print(f"  {state}: {count} occurrences ({count/len(states)*100:.1f}%)")
    
    return states, unique_states

def build_transition_matrix(states, k_order=1):
    """Build k-order Markov chain transition matrix"""
    if k_order < 1:
        raise ValueError("k_order must be at least 1")
    
    if len(states) <= k_order:
        raise ValueError(f"Need more than {k_order} data points for {k_order}-order Markov chain")
    
    # Count transitions
    transitions = defaultdict(Counter)
    
    for i in range(len(states) - k_order):
        # Current state sequence (k previous states)
        current_sequence = tuple(states[i:i+k_order])
        # Next state
        next_state = states[i+k_order]
        # Count this transition
        transitions[current_sequence][next_state] += 1
    
    # Convert to probabilities
    transition_probs = {}
    for sequence, next_states in transitions.items():
        total = sum(next_states.values())
        transition_probs[sequence] = {
            state: count/total for state, count in next_states.items()
        }
    
    print(f"Built {k_order}-order transition matrix with {len(transition_probs)} state sequences")
    
    return transition_probs

def predict_next_state(recent_states, transition_matrix, all_states, k_order):
    """Predict the next state given recent states"""
    if len(recent_states) != k_order:
        raise ValueError(f"Need exactly {k_order} recent states, got {len(recent_states)}")
    
    current_sequence = tuple(recent_states)
    
    # Try exact match first
    if current_sequence in transition_matrix:
        probs = transition_matrix[current_sequence]
        # Get the most likely next state
        predicted_state = max(probs.keys(), key=probs.get)
        probability = probs[predicted_state]
        return predicted_state, probability, "exact match"
    
    # Fallback strategies for unseen sequences
    
    # Try shorter sequences (order reduction)
    for order in range(k_order-1, 0, -1):
        partial_sequence = current_sequence[-order:]
        matching_sequences = [seq for seq in transition_matrix.keys() 
                            if seq[-order:] == partial_sequence]
        
        if matching_sequences:
            # Combine probabilities from all matching sequences
            combined_probs = defaultdict(float)
            total_weight = 0
            
            for seq in matching_sequences:
                weight = sum(transition_matrix[seq].values())  # frequency weight
                total_weight += weight
                for state, prob in transition_matrix[seq].items():
                    combined_probs[state] += prob * weight
            
            # Normalize
            for state in combined_probs:
                combined_probs[state] /= total_weight
            
            predicted_state = max(combined_probs.keys(), key=combined_probs.get)
            probability = combined_probs[predicted_state]
            return predicted_state, probability, f"partial match (order {order})"
    
    # Last resort: most common state overall
    all_next_states = []
    for transitions in transition_matrix.values():
        for state, count in transitions.items():
            all_next_states.extend([state] * int(count * 100))  # approximate frequency
    
    if all_next_states:
        most_common = Counter(all_next_states).most_common(1)[0]
        return most_common[0], most_common[1]/len(all_next_states), "most common fallback"
    
    # Ultimate fallback
    return all_states[0], 0.1, "random fallback"

def predict_from_custom_input(custom_changes, transition_matrix, all_states, k_order):
    """Predict next state from custom daily change inputs"""
    if len(custom_changes) != k_order:
        raise ValueError(f"Need exactly {k_order} daily change values")
    
    # Convert custom changes to states
    def get_state(change):
        if change <= -2.0:
            return "Large Drop (-2%+)"
        elif change <= -1.0:
            return "Drop (-1% to -2%)"
        elif change <= -0.5:
            return "Small Drop (-0.5% to -1%)"
        elif change <= 0.0:
            return "Slight Drop (0% to -0.5%)"
        elif change <= 0.5:
            return "Slight Rise (0% to 0.5%)"
        elif change <= 1.0:
            return "Small Rise (0.5% to 1%)"
        elif change <= 2.0:
            return "Rise (1% to 2%)"
        else:
            return "Large Rise (2%+)"
    
    custom_states = [get_state(change) for change in custom_changes]
    
    # Use the regular prediction function
    return predict_next_state(custom_states, transition_matrix, all_states, k_order), custom_states

def get_state_center_value(state):
    """Get the center value of a state range for numerical prediction"""
    if "Large Drop" in state:
        return -2.5
    elif "Drop (-1% to -2%)" in state:
        return -1.5
    elif "Small Drop (-0.5% to -1%)" in state:
        return -0.75
    elif "Slight Drop" in state:
        return -0.25
    elif "Slight Rise" in state:
        return 0.25
    elif "Small Rise" in state:
        return 0.75
    elif "Rise (1% to 2%)" in state:
        return 1.5
    elif "Large Rise" in state:
        return 2.5
    else:
        return 0.0
