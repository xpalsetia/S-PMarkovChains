from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from markov_chain import build_transition_matrix, predict_next_state, load_data

app = Flask(__name__)

# Load and preprocess the CSV data
df = load_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            # Get the inputs from the form
            k_value = int(request.form['k_value'])
            daily_change_input = float(request.form['daily_change'])

            # Build transition matrices
            transitions = build_transition_matrix(df, k_value)

            # Find the closest state to the given daily change
            states = df['State'].unique()
            closest_state = min(states, key=lambda x: abs(x - daily_change_input))
            state_map = {state: idx for idx, state in enumerate(states)}
            current_state_id = state_map[closest_state]

            # Get the most recent k values from the data to base the prediction on
            recent_states = [state_map[df.iloc[-i]['State']] for i in range(1, k_value+1)]

            # Predict the next day's state
            predicted_change, probability = predict_next_state(recent_states, transitions, k_value)

            prediction = f"Predicted percentage change for tomorrow: {predicted_change}%"
            probability = f"Probability of this prediction: {probability:.4f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
