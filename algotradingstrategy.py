import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import os
import json

# Set environment variables for simulation. In production, these will be set in the AWS environment.
os.environ['PARAMS_FILE'] = 'kalman_params.json'
os.environ['API_KEY'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['API_SECRET'] = os.getenv('AWS_SECRET_ACCESS_KEY')

def load_state():
    params_file = os.environ.get('PARAMS_FILE', 'kalman_params.json')
    try:
        with open(params_file, 'r') as file:
            state = json.load(file)
        params = np.array(state['params'])
        last_residual = state['last_residual']
        print("Cache file loaded successfully.")
    except FileNotFoundError:
        # Default parameters
        params = np.array([0.3, 0.9, 0.8, 1.1])
        last_residual = 0
        print("No cache file found. Using default parameters.")
    except json.JSONDecodeError:
        # Handling JSON decode error (corrupted file, etc.)
        params = np.array([0.3, 0.9, 0.8, 1.1])
        last_residual = 0
        print("Corrupted cache file. Using default parameters.")
    return params, last_residual

def save_state(params, last_residual):
    params_file = os.environ.get('PARAMS_FILE', 'kalman_params.json')
    state = {'params': params.tolist(), 'last_residual': last_residual}
    with open(params_file, 'w') as file:
        json.dump(state, file)
    print("State saved to cache.")

def fetch_data():
    start_date = datetime(2020, 1, 1)
    end_date = date.today()
    data = yf.download('META', start=start_date, end=end_date)
    data['Typical_Price'] = data[['High', 'Low', 'Close']].mean(axis=1)
    data['lrets'] = (np.log(data['Close']) - np.log(data['Close'].shift(1))) * 100
    data.dropna(subset=['lrets'], inplace=True)
    print("Data fetched from Yahoo Finance.")
    return data

def kalman_filter(Y, params):
    S = len(Y)
    Z, T, H, Q = params
    u_predict = np.zeros(S + 1)
    P_predict = np.zeros(S + 1)
    u_update = np.zeros(S + 1)
    P_update = np.zeros(S + 1)
    P_update[0] = 1000
    
    residuals = np.zeros(S)
    
    for t in range(1, S + 1):
        F = Z * P_predict[t - 1] * Z + H
        v = Y[t - 1] - Z * u_predict[t - 1]
        residuals[t - 1] = v
        u_update[t] = u_predict[t - 1] + P_predict[t - 1] * Z * (1/F) * v
        P_update[t] = P_predict[t - 1] - P_predict[t - 1] * Z * (1/F) * Z * P_predict[t - 1]
        u_predict[t] = T * u_update[t]
        P_predict[t] = T * P_update[t] * T + Q
    
    last_residual = residuals[-1]
    print(f"Kalman filter applied. Last residual: {last_residual}")
    return u_update[-1], last_residual

def run_model():
    data = fetch_data()
    params, last_residual = load_state()
    prediction, last_residual = kalman_filter(data['Close'].values, params)
    save_state(params, last_residual)
    
    latest_close_price = data['Close'].iloc[-1]
    print(f"Latest Close Price: {latest_close_price}")
    print(f"Predicted Next Close Price: {prediction}")
    action = 'Buy' if prediction > latest_close_price else 'Sell'
    print(f"Recommended Action for Tomorrow: {action}")
    return {
        "Latest Close Price": latest_close_price,
        "Predicted Next Close Price": prediction,
        "Action": action
    }

# This function is intended to be used in an AWS Lambda to trigger predictions
def lambda_handler(event, context):
    return run_model()

if __name__ == '__main__':
    # Local test run
    result = run_model()
    print(result)
