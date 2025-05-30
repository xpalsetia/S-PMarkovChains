from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from markov_chain import (load_data, create_states, build_transition_matrix, 
                         predict_next_state, get_state_center_value, predict_from_custom_input)

app = Flask(__name__)

# Global variables to store data
data = None
states = None
unique_states = None

def initialize_data():
    """Initialize the data when the app starts"""
    global data, states, unique_states
    try:
        data = load_data()
        states, unique_states = create_states(data)
        print("Data initialized successfully")
    except Exception as e:
        print(f"Error initializing data: {e}")
        data, states, unique_states = None, None, None

# Initialize data on startup
initialize_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    method_used = None
    recent_states_info = None
    custom_states_info = None
    error = None

    if request.method == 'POST':
        if data is None or states is None:
            error = "Data not loaded. Please check if s&p500_data.csv exists and is properly formatted."
        else:
            try:
                # Get the k-order from the form
                k_value = int(request.form.get('k_value', 2))
                
                # Check which prediction mode to use
                prediction_mode = request.form.get('prediction_mode', 'historical')
                
                # Validate k_value
                if k_value < 1:
                    raise ValueError("K-value must be at least 1")
                if k_value > 10:
                    raise ValueError("K-value should not exceed 10 for practical purposes")
                if k_value >= len(states):
                    raise ValueError(f"K-value must be less than the number of data points ({len(states)})")

                # Build the transition matrix
                transition_matrix = build_transition_matrix(states, k_value)
                
                if not transition_matrix:
                    raise ValueError("Could not build transition matrix - insufficient data")

                if prediction_mode == 'custom':
                    # Get custom daily changes from form
                    custom_changes = []
                    for i in range(k_value):
                        change_key = f'custom_change_{i}'
                        change_value = request.form.get(change_key)
                        if change_value is None or change_value.strip() == '':
                            raise ValueError(f"Please enter all {k_value} daily change values")
                        try:
                            custom_changes.append(float(change_value))
                        except ValueError:
                            raise ValueError(f"Invalid number format for daily change {i+1}")
                    
                    # Validate custom changes (reasonable range)
                    for i, change in enumerate(custom_changes):
                        if abs(change) > 20:  # 20% seems like a reasonable max for daily changes
                            raise ValueError(f"Daily change {i+1} seems too large ({change}%). Please enter a reasonable value.")
                    
                    # Predict from custom input
                    (predicted_state, prob, method), custom_states = predict_from_custom_input(
                        custom_changes, transition_matrix, unique_states, k_value
                    )
                    
                    custom_states_info = f"Your input states: " + " → ".join(custom_states)
                    recent_states_info = f"Daily changes: {' → '.join([f'{c:+.2f}%' for c in custom_changes])}"
                    
                else:
                    # Use historical data (original functionality)
                    recent_states = states[-k_value:]
                    recent_states_info = f"Recent {k_value} states: " + " → ".join(recent_states)

                    # Predict the next state
                    predicted_state, prob, method = predict_next_state(
                        recent_states, transition_matrix, unique_states, k_value
                    )

                # Convert state back to numerical value
                predicted_value = get_state_center_value(predicted_state)

                prediction = f"Predicted state: {predicted_state}"
                probability = f"Confidence: {prob:.3f} ({prob*100:.1f}%)"
                method_used = f"Prediction method: {method}"

            except ValueError as ve:
                error = f"Input Error: {str(ve)}"
            except Exception as e:
                error = f"Calculation Error: {str(e)}"

    # Prepare template data
    template_data = {
        'prediction': prediction,
        'probability': probability,
        'method_used': method_used,
        'recent_states_info': recent_states_info,
        'custom_states_info': custom_states_info,
        'error': error,
        'data_loaded': data is not None,
        'total_data_points': len(data) if data is not None else 0,
        'data_range': f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}" if data is not None else "N/A"
    }

    return render_template('index.html', **template_data)

@app.route('/data_info')
def data_info():
    """Endpoint to show data information"""
    if data is None:
        return "No data loaded"
    
    info = {
        'total_points': len(data),
        'date_range': f"{data['Date'].min()} to {data['Date'].max()}",
        'change_stats': {
            'min': float(data['Change Rounded'].min()),
            'max': float(data['Change Rounded'].max()),
            'mean': float(data['Change Rounded'].mean()),
            'std': float(data['Change Rounded'].std())
        },
        'states': unique_states,
        'state_distribution': {state: states.count(state) for state in unique_states}
    }
    
    return info

if __name__ == '__main__':
    if data is not None:
        print("App starting with data loaded successfully")
        print(f"Data points: {len(data)}")
        print(f"States: {len(unique_states)}")
    else:
        print("App starting without data - check your CSV file")
    
    app.run(debug=True)
