#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, learning_curve, train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def fit_model(X, y):

    cv_sets = ShuffleSplit(test_size = 0.20, random_state = 0)

    params = {'n_estimators':[100, 120, 140],
              'min_samples_leaf':[1, 2, 3],
              'max_depth':list(range(1, 20)),
              'max_features':[0.05, 0.1, 0.15, 0.2]}

    n_iter_search = 20
    regressor = RandomForestRegressor()
    score = make_scorer(r2_score)
    grid = RandomizedSearchCV(regressor, params, n_iter = n_iter_search, scoring = score, cv = cv_sets)
    grid = grid.fit(X, y)
    # Return the optimal model
    return grid.best_estimator_

def home():
    dataset = pd.read_csv('data-new.csv')

    dataset.index =  pd.to_datetime(dataset['Date'])

    dataset['MonthOfYear'] = dataset.index.strftime('%m').astype(int)
    dataset['DayOfYear'] = dataset.index.strftime('%j').astype(int)
    dataset['WeekOfYear'] = dataset.index.strftime('%U').astype(int)
    dataset.drop(['Date'], inplace=True, axis=1)

    dataset = dataset.dropna()

    X = dataset[['Station Id', 'Air Temperature Maximum (degF)',
       'Air Temperature Minimum (degF)', 'Precipitation Increment (in)',
       'Relative Humidity (pct) Mean of Hourly Values',
       'Wind Speed Maximum (mph) Max of Hourly Values',
       'Wind Speed Average (mph) Mean of Hourly Values',
       'Solar Radiation Average (watt/m2) Mean of Hourly Values',
       'Solar Radiation/langley Total (langley)',
       'Vapor Pressure - Partial (inch_Hg) Mean of Hourly Values',
       'Vapor Pressure - Saturated (inch_Hg) Mean of Hourly Values','Soil Temperature Observed -2in (degF) Mean of Hourly Values',
       'Soil Temperature Observed -4in (degF) Mean of Hourly Values',
       'Soil Temperature Observed -8in (degF) Mean of Hourly Values',
       'Soil Temperature Observed -20in (degF) Mean of Hourly Values',
       'Soil Temperature Observed -40in (degF) Mean of Hourly Values']]
    y = dataset['Soil Moisture Percent -2in (pct) Mean of Hourly Values']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    regressor = RandomForestRegressor(n_estimators = 100)

        # if request.method == 'POST':
        #     int_features = [int(x) for x in request.form.values()]
        #     final_features = [np.array(int_featsures)]
    reg = fit_model(X_train, y_train)

    y_opt_pred = reg.predict(X_test)
    prediction = regressor.predict([[34,13,16]])

    return render_template('index.html', temp=temp, pressure= pressure, humidity= humidity, date=date, wind= wind, prediction=round(prediction[0],3))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    dataset = pd.read_csv('data/SI.csv')
    X = dataset[['Temperature','DayOfYear', 'TimeOfDay(s)']]
    y = dataset['Radiation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    regressor = RandomForestRegressor(n_estimators = 100)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # if request.method == 'POST':
    #     int_features = [int(x) for x in request.form.values()]
    #     final_features = [np.array(int_featsures)]
    prediction = regressor.predict([[34,13,16]])

    return render_template('index.html', predictions=prediction)

if __name__ == "__main__":
    app.run(port=8060, debug=True)
