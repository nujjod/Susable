#!/usr/bin/env python
# coding: utf-8

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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')

import warnings
warnings.filterwarnings("ignore")

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

def ModelLearning(X, y):

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    fig = plt.figure(figsize=(15,10))

    for k, depth in enumerate([1,3,6,10]):

        # Create a random forest regressor
        regressor = RandomForestRegressor(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = learning_curve(regressor, X, y,             cv = cv, train_sizes = train_sizes, scoring = 'r2')

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'purple', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std,             train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std,             test_mean + test_std, alpha = 0.15, color = 'purple')

        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Random Forest Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()

regressor = RandomForestRegressor(n_estimators = 100)

from sklearn.model_selection import validation_curve

def ModelComplexity(X, y):

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(RandomForestRegressor(), X, y,         param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(15,10))
    plt.title('Random Forest Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth, train_mean - train_std,         train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth, test_mean - test_std,         test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()

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

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

y_opt_pred = reg.predict(X_test)

mean_squared_error = mean_squared_error(y_test, y_opt_pred)
r_squared = r2_score(y_test, y_opt_pred)
print('mse = {}'.format(mean_squared_error))

print('r2 = {}'.format(round(r_squared*100)))
