from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras import layers, models
import keras

from src.helper_functions.training_functions import train_lm, train_nn, train_rf
from src.helper_functions.data_prep import split_xy, add_shift_predictors

'''
This script trains all 3 models, then evaluates on the test set and produces the scores for all hyperparameters and 
requires data/derived/cleaned_df.csv
The output are csv dataframes that contain the results which can be used for plotting later:
    output/lm_results.csv
    output/rf_results.csv
    output/nn_results.csv
    output/test_scores.csv
We have also added a check on whether results have already been previously generated, so that it skips those.
This allows for shorter run time in the case where for example 1 model has already been trained and result produced, but
other model results have not been produced yet.
'''
no_na_df = pd.read_csv('data/derived/cleaned_df.csv')

time_dep_variables = ['people_fully_vaccinated_per_hundred', 'new_cases_smoothed_per_million', 'tests_per_case',
            'new_vaccinations_smoothed_per_million', 'stringency_index']
response_variables = ['new_deaths_smoothed_per_million', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions_per_million']
const_variables = ['population_density', 'median_age', 'aged_65_older',
                  'aged_70_older', 'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence',
                  'male_smokers', 'female_smokers', 'hospital_beds_per_thousand',
                  'life_expectancy', 'human_development_index']

# Split the data into train and test sets
train_df = no_na_df.groupby('iso_code', group_keys=False).apply(lambda group: group.iloc[:9*len(group.index)//10,:])
test_df = no_na_df.groupby('iso_code', group_keys=False).apply(lambda group: group.iloc[9*len(group.index)//10:,:])

# Rescale dataframes into [0,1]
x_scaler = MinMaxScaler()
train_df[time_dep_variables+const_variables] = x_scaler.fit_transform(train_df[time_dep_variables+const_variables])
y_scaler = MinMaxScaler()
train_df[response_variables] = y_scaler.fit_transform(train_df[response_variables])

test_df[time_dep_variables+const_variables] = x_scaler.transform(test_df[time_dep_variables+const_variables])
test_df[response_variables] = y_scaler.transform(test_df[response_variables])

'''
Training the linear model
'''

alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2]
reg_type_list = ['lasso','ridge']
shift_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results_path = 'output/lm_results.csv'
if os.path.exists(results_path):
    print('Linear Model results are already produced.')
else:
    results_scores = []
    for alpha in alpha_list:
        for reg_type in reg_type_list:
            for shift in shift_list:
                scores = train_lm(train_df, shift=shift, reg_type=reg_type, alpha=alpha)
                mean_score = np.mean(scores)
                results_scores.append({
                'shift': shift,
                'regularisation type': reg_type,
                'regularisation magnitude': alpha,
                'validation score': mean_score
                })

    # Convert scores to DataFrame
    results_df = pd.DataFrame(results_scores)
    results_df.to_csv(results_path)

'''
Training the random forest model
'''
max_depth_list = [5, 10, 20, 40, None]
bootstrap_list = [True, False]
shift_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results_path = 'output/rf_results.csv'
if os.path.exists(results_path):
    print('Random Forest Model results are already produced.')
else:
    results_scores = []
    for max_depth in max_depth_list:
        for bootstrap in bootstrap_list:
            for shift in shift_list:
                scores = train_rf(train_df, shift=shift, bootstrap=bootstrap, max_depth=max_depth)
                mean_score = np.mean(scores)
                results_scores.append({
                'shift': shift,
                'max depth': max_depth,
                'bootstrap': bootstrap,
                'validation score': mean_score
                })

    # Convert scores to DataFrame
    results_df = pd.DataFrame(results_scores)
    results_df.to_csv(results_path)

'''
Training the MLP model
'''

layer_list = [1, 2, 3]
unit_list = [256, 512, 1024]
lr = 0.0001
shift_list = [1, 5, 10]

results_path = 'output/nn_results.csv'
if os.path.exists(results_path):
    print('Random Forest Model results are already produced.')
else:
    results_scores = []
    for layer in layer_list:
        for unit in unit_list:
            for shift in shift_list:
                # Define model based on hyperparameters
                model = models.Sequential()
                model.add(layers.Input(shape=(12+8*shift,)))
                for _ in range(layer):
                    model.add(layers.Dense(unit, activation='relu'))
                    model.add(layers.Dropout(0.1))
                model.add(layers.Dense(3, activation='sigmoid'))
                
                dir_path = f'output/nn/{layer}layer_{unit}unit_{shift}shift'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                model, val_score = train_nn(model, train_df, shift=shift, epochs=50,
                                    chkpt_path=dir_path+'/backup_chkpt', best_path=dir_path+'/best_chkpt.keras', log_path=dir_path+'/test_log.csv', 
                                    patience=5, lr=0.001, img_path=dir_path+'/loss.png')
                results_scores.append({
                'shift': shift,
                'units per layer': unit,
                'number of hidden layers': layer,
                'validation score': val_score.numpy()
                })
    # Convert scores to DataFrame
    results_df = pd.DataFrame(results_scores)
    results_df.to_csv(results_path)


'''
Evaluate Model on Test Set for Linear Model
'''
if os.path.exists('output/test_scores.csv'):
    print('Score results are already produced.')
else:
    # Evaluate linear model
    results_path = 'output/lm_results.csv'
    results_df = pd.read_csv(results_path)

    # Get best hyperparameters
    best = results_df.iloc[results_df['validation score'].idxmax(), :]
    shift = best['shift']
    type = best['regularisation type']
    alpha = best['regularisation magnitude']
    val_score = best['validation score']

    lag_train_df = add_shift_predictors(train_df, shift)
    lag_test_df = add_shift_predictors(test_df, shift)

    # Drop irrelevant columns to get x,y
    x_train, y_train = split_xy(lag_train_df)
    x_test, y_test = split_xy(lag_test_df)

    if type == 'lasso':
        model = linear_model.LassoLars(alpha=alpha)
    else:
        model = linear_model.Ridge(alpha=alpha)

    model.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)

    # Collate test scores
    all_models_results = []

    all_models_results.append({
                    'Model Type': f'Linear model; shift {shift}; regularisation type {type}; alpha {alpha}',
                    'Validation Score': val_score,
                    'Test Score': test_score
                    })

    '''
    Evaluate for Random Forest
    '''
    # Evaluate Random Forest Model
    results_path = 'output/rf_results.csv'
    results_df = pd.read_csv(results_path)

    best = results_df.iloc[results_df['validation score'].idxmax(), :]
    shift = best['shift']
    max_depth = int(best['max depth'])
    bootstrap = best['bootstrap']
    val_score = best['validation score']

    lag_train_df = add_shift_predictors(train_df, shift)
    lag_test_df = add_shift_predictors(test_df, shift)

    # Drop irrelevant columns to get x,y
    x_train, y_train = split_xy(lag_train_df)
    x_test, y_test = split_xy(lag_test_df)

    model = RandomForestRegressor(max_depth=max_depth, bootstrap=bootstrap, random_state=1843091, n_jobs=-1)
    model.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)

    all_models_results.append({
                    'Model Type': f'Random Forest model; shift {shift}; max depth {max_depth}; bootstrap {bootstrap}',
                    'Validation Score': val_score,
                    'Test Score': test_score
                    })


    '''
    Evaluate for MLP model
    '''
    results_path = 'output/nn_results.csv'
    results_df = pd.read_csv(results_path)

    best = results_df.iloc[results_df['validation score'].idxmax(), :]
    shift = best['shift'].astype(np.int32)
    unit = best['units per layer'].astype(np.int32)
    layer = best['number of hidden layers'].astype(np.int32)
    val_score = best['validation score']
    print(shift, unit, layer)


    # Create lag features for predictors
    lag_train_df = add_shift_predictors(train_df, shift)
    lag_test_df = add_shift_predictors(test_df, shift)

    # Drop irrelevant columns to get x,y
    x_train, y_train = split_xy(lag_train_df)
    x_test, y_test = split_xy(lag_test_df)

    # Define model based on hyperparameters
    model = models.Sequential()
    model.add(layers.Input(shape=(12+8*shift,)))
    for _ in range(layer):
        model.add(layers.Dense(unit, activation='relu'))
        model.add(layers.Dropout(0.1))
    model.add(layers.Dense(3, activation='sigmoid'))

    weight_path = f'output/nn/{layer}layer_{unit}unit_{shift}shift/best_chkpt.keras'
    model.load_weights(weight_path)

    y_pred = model.predict(x_test)
    scorer = keras.metrics.R2Score()
    scorer.update_state(tf.cast(y_test, tf.float32), y_pred)

    test_score = scorer.result()

    all_models_results.append({
                    'Model Type': f'MLP model; shift {shift}; {layer} hidden layers; {unit} units per layer',
                    'Validation Score': val_score,
                    'Test Score': test_score.numpy()
                    })

    # Save test scores
    results_df = pd.DataFrame(all_models_results)
    results_df.to_csv('output/test_scores.csv')