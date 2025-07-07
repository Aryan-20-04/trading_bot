import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----- User Inputs -----
position_size = float(input("Enter the percentage of portfolio to risk per trade (e.g., 0.02 for 2%): "))
stop_loss_percentage = float(input("Enter the stop-loss percentage (e.g., 0.03 for 3%): "))

# ----- Download Data -----
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data['MA10'] = data['Close'].rolling(window=10).mean()
data = data.dropna()

features = data[['MA10']]
target = data['Close'].shift(-1).dropna()
features = features.iloc[:-1]  # Align target with features

# ----- Scaling -----
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ----- Hyperparameter Tuning -----
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'batch_size': [16, 32],
    'max_iter': [3000, 4000],
    'tol': [1e-4, 1e-3]
}
test_model = MLPRegressor(random_state=42)
grid_search = GridSearchCV(test_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)

try:
    grid_search.fit(features_scaled, target)
    model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Score (Negative MSE): {grid_search.best_score_}")

    # ----- Cross-validated Error -----
    mse_list = []
    for train_idx, test_idx in tscv.split(features_scaled):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mse_list.append(mse)

    print(f"Average Mean Squared Error: {np.mean(mse_list)}")

    # ----- Make Predictions -----
    data = data.iloc[:-1]  # Align with features
    data['Prediction'] = model.predict(features_scaled)
    data['Signal'] = data['Prediction'] > data['Close']

    # ----- Buy and Hold -----
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100

    # ----- Simulated Strategy -----
    portfolio_val = 100000
    positions = []
    stop_losses = []

    for i in range(1, len(data)):
        close_price = data['Close'].iloc[i]

        # Entry Signal
        if data['Signal'].iloc[i] and portfolio_val > 0:
            position_size_val = portfolio_val * position_size
            stop_loss_val = close_price * (1 - stop_loss_percentage)
            positions.append({
                'Date': data.index[i],
                'Entry Price': close_price,
                'Position Size': position_size_val,
                'Stop-loss': stop_loss_val
            })
            stop_losses.append(stop_loss_val)

        # Stop Loss Exit
        if positions and close_price < stop_losses[-1]:
            p = positions.pop()
            stop_losses.pop()
            exit_val = close_price
            gain = p['Position Size'] * (exit_val / p['Entry Price'] - 1)
            portfolio_val += gain

    # Close remaining positions at last price
    for p in positions:
        final_val = p['Position Size'] * (data['Close'].iloc[-1] / p['Entry Price'])
        portfolio_val += final_val

    strategy_return = (portfolio_val / 100000 - 1) * 100

    # ----- Output -----
    print(f"\nBuy and Hold Returns: {buy_hold_return:.2f}%")
    print(f"Strategy Returns with Risk Management: {strategy_return:.2f}%")

except KeyboardInterrupt:
    print("Training interrupted by user.")
    model = None
