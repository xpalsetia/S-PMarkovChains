# S&P 500 Markov Chain Prediction

This project uses higher-order Markov Chains to predict the daily percentage change in the S&P 500 index based on historical data. It allows users to input the **k-value** (order of the Markov Chain) and a daily percentage change value, and it predicts the next day's market movement along with the probability of that prediction.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/xpalsetia/S-PMarkovChains.git
    cd S-PMarkovChains
    ```

2. **Install dependencies**:

    Use `pip` to install the required Python libraries:

    ```bash
    pip install flask pandas numpy
    ```

3. **Prepare the CSV file**:

    Update CSV data (named `s&p500_data.csv`) in the project directory. The CSV should contain the following columns:
    
    ```
    Date, Daily Change, Change Rounded, Close/Last, Open, High, Low
    ```

    The `Change Rounded` column should be the percentage change in the market.

4. **Running the Application**:

    Start the Flask server by running the following command:

    ```bash
    python3 app.py
    ```

    This will start the Flask app at `http://127.0.0.1:5000/`.

5. **Access the Web Interface**:

    Open a browser and go to `http://127.0.0.1:5000/` to use the web interface.

    - Enter the **k-value** (order of the Markov Chain) and the **daily percentage change**.
    - The app will predict the next day's percentage change and show the associated probability.

## How It Works

1. **Markov Chain Model**:

   - The app uses a **higher-order Markov Chain** to model the daily price changes. The user can choose the order of the Markov Chain (k-value), which influences how many previous days' changes are used to predict the next day's change.
   - Transition matrices are built for the given k-order, and these matrices are used to predict the next state (percentage change).

2. **Prediction Process**:

   - The Markov Chain predicts the percentage change for the next day based on the most recent k-day percentage changes.
   - The app then calculates the probability of this prediction.

3. **Data Handling**:

   - The application reads the historical S&P 500 data from a CSV file and processes the `Change Rounded` column. If any invalid data (like `#DIV/0!`) is encountered, it is replaced with `0`.
   - The data is cleaned and formatted for use in the Markov Chain model.

## Example

If you enter a k-value of 3 and a daily change of `0.5%`, the app will:

1. Look at the previous 3 days' changes.
2. Use the transition matrix for a 3rd-order Markov Chain to predict the next day's percentage change.
3. Display the predicted change and the probability of the prediction.

## Technologies Used

- **Python 3**: Programming language.
- **Flask**: Web framework for creating the web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and matrix handling.
- **HTML/CSS**: For front-end web interface.

## Future Improvements

- **Add More Features**: Incorporate additional market features like volume, sentiment analysis, or macroeconomic data.
- **Model Tuning**: Experiment with different order Markov Chains and other models like HMM (Hidden Markov Models) or machine learning approaches to improve predictions.
- **Fix Live Data Mode**: Get API working for that
