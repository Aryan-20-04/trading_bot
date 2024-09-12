import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

position_size = float(input("Enter the percentage of portfolio to risk per trade (e.g., 0.02 for 2%): "))
stop_loss_percentage = float(input("Enter the stop-loss percentage (e.g., 0.03 for 3%): "))

data=yf.download('AAPL',start='2020-01-01',end='2023-01-01')
data['MA10']=data['Close'].rolling(window=10).mean()
data=data.dropna()

features=data[['MA10']]
target=data['Close'].shift(-1).dropna()
features=features.iloc[:-1]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

tscv=TimeSeriesSplit(n_splits=5)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'batch_size': [16,32],
    'max_iter': [3000,4000],
    'tol':[1e-4,1e-3]
}
test_model=MLPRegressor(random_state=42)
grid_search = GridSearchCV(estimator=test_model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
try:
    grid_search.fit(features_scaled, target)
    model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score (Negative MSE): {grid_search.best_score_}")

    mse_list=[]

    for train_index,test_index in tscv.split(features_scaled):
        X_train,X_test=features_scaled.iloc[train_index],features_scaled[test_index]
        y_train,y_test=target.iloc[train_index],target.iloc[test_index]

        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        mse=mean_squared_error(y_test,pred)
        mse_list.append(mse)
    
    mean_mse=np.mean(mse_list)
    print(f"AVerage Mean Squared Error: {mean_mse}")
    data['Prediction']=np.nan
    data['Signal']=False

    data.loc[features.index,'Prediction']=model.predict(features_scaled)
    data.loc[features.index,'Signal']=data['Prediction']>data['Close']

    buy_hold=data['Close'].pct_change().cumsum()

    #risk management
    portfolio_val=100000
    position=[]
    stop_loss=[]
    for i in range(1,len(data)):
        if data['Signal'].iloc[i] and portfolio_val>0:
            position_size_val=portfolio_val*position_size
            stop_loss_val=data['Close'].iloc[i]*(1-stop_loss_percentage)
            
            position.append({'Date':data.index[i],'Entry Price':data['Close'].iloc[i],'Position Size':position_size_val,'Stop-loss':stop_loss_val})
            stop_loss.append(stop_loss_val)
            
        if position and data['Close'].iloc[i]<stop_loss[-1]:
            p=position.pop()
            exit_val=data['Close'].iloc[i]
            portfolio_val+=p['Position_size']*(exit_val/p['Entry Price']-1)
            stop_loss.pop()
            
    final_portfolio_value = portfolio_val + sum(p['Position Size'] * (data['Close'].iloc[-1] / p['Entry Price'] - 1) for p in position)
    strat_return = (final_portfolio_value / 100000 - 1) * 100

    # Print final returns
    print(f'Buy and Hold Returns: {buy_hold.iloc[-1] * 100:.2f}%')
    print(f'Strategy Returns with Risk Management: {strat_return:.2f}%')
except KeyboardInterrupt:
    print("Training interupted by user")
    model=None