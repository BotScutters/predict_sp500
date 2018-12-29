# Imports
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data import and prep
data = pd.read_csv('sphist.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', ascending=True, inplace=True)


## Compute indicators
# 5, 30, and 365 day moving averages
data['5MA'] = data['Close'].rolling(5).mean().shift()
data['30MA'] = data['Close'].rolling(30).mean().shift()
data['365MA'] = data['Close'].rolling(365).mean().shift()

# Ratio of 5MA to 365MA
data['5MA/365MA'] = data['5MA'] / data['365MA']

# StDev of 5MA and 365MA
data['5stdev'] = data['Close'].rolling(5).std().shift()
data['365stdev'] = data['Close'].rolling(365).std().shift()

# Ratio of 5stdev to 365stdev
data['5/365stdev'] = data['5stdev'] / data['365stdev']

# Year, month, and day components of the date
data['year'] = data['Date'].map(lambda x: x.year)
data['month'] = data['Date'].map(lambda x: x.month)
data['month_day'] = data['Date'].map(lambda x: x.day)

# 1, 16, and 256 day deltas (close - close n days before)
deltas = [1, 16, 256]
for n in deltas:
    data['delta' + str(n)] = data['Close'].diff(periods=n).shift()
#data['delta_1'] = data['Close'].diff(periods=1)
#data['delta_16'] = data['Close'].diff(periods=16)
#data['delta_256'] = data['Close'].diff(periods=256)

## We've got NaNs now thanks to our moving averages. Drop em!
data.dropna(axis=0, inplace=True)

## Generate training and testing sets
cutoff_date = '2013-01-01'
train = data[data['Date'] < cutoff_date]
test = data[data['Date'] >= cutoff_date]

## Regression
def regress(train, test):
    '''
    Function to fit, and predict with a linear regression model;
    returns model and RMSE
    columns:
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
        '5MA', '30MA', '365MA', '5MA/365MA',
        '5stdev', '365stdev', '5/365stdev', 
        'year', 'month', 'month_day',
        'delta1', 'delta16', 'delta256'
    
    Define features and target
    '''
    features = ['5MA', '30MA', '365MA', '5MA/365MA',
                '5stdev', '365stdev', '5/365stdev', 
                'year', 'month', 'month_day',
                'delta1', 'delta16', 'delta256']
    target = 'Close'
    
    # Train model
    reg = LinearRegression().fit(train[features], train[target])
    y_pred = reg.predict(test[features])
        
    # Calculate RMSE
    rmse = mean_squared_error(test[target], y_pred)
    return rmse

rmse = regress(train, test)

print(rmse)
