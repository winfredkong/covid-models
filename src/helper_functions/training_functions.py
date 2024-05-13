from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, BackupAndRestore, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.helper_functions.data_prep import add_shift_predictors, split_xy


'''
This script defines training functions for all 3 model types
'''

def train_lm(df, shift, reg_type, alpha, n_splits=5):
    '''
    Trains a linear regression model on df using walk-forward validation and hyperparameter choice
    
    Inputs:
    df is the dataframe which should have all the columns of original dataset
    shift is an integer representing the number of days in the past to use as forecast
    reg_type should be a string with 'lasso' represeting LASSO regularisation
        or 'ridge' for ridge regularisation
    alpha is the float representing the magnitude of regularisation
    n_splits is the number of splits (note that this means the technically the data is split into
        n_splits+1 pieces) for walk-forward validation

    Returns:
    scores a list (of length n_splits) containing the R^2 validation scores of all splits
    '''
    lag_df = add_shift_predictors(df, shift)
    
    scores = []
    # Walk-forward validation
    for i in range(n_splits):
        # Get the different blocks of data for training and validation
        train_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[:(i+1)*len(group.index)//(n_splits+1),:])
        val_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[(i+1)*len(group.index)//(n_splits+1):(i+2)*len(group.index)//(n_splits+1),:])

        # Drop irrelevant columns to get x,y
        x_train, y_train = split_xy(train_df)
        x_val, y_val = split_xy(val_df)

        if reg_type == 'lasso':
            model = linear_model.LassoLars(alpha=alpha)
            model.fit(x_train, y_train)
            split_score = model.score(x_val, y_val)
        elif reg_type == 'ridge':
            model = linear_model.Ridge(alpha=alpha)
            model.fit(x_train, y_train)
            split_score = model.score(x_val, y_val)
        scores.append(split_score)
        
    return scores



def train_rf(df, shift, max_depth, bootstrap, seed=1843091, n_splits=5):
    '''
    Trains a random forest model on df using walk-forward validation and hyperparameter choice
    
    Inputs:
    df is the dataframe which should have all the columns of original dataset
    shift is an integer representing the number of days in the past to use as forecast
    max_depth is an integer representing the maximum depth of the tree. Can also be set to 
        None if no maximum depth
    bootstrap is a boolean representing whether to bootstrap samples of data 
    n_splits is the number of splits (note that this means the technically the data is split into
        n_splits+1 pieces) for walk-forward validation

    Returns:
    scores a list (of length n_splits) containing the R^2 validation scores of all splits
    '''
    lag_df = add_shift_predictors(df, shift)
    
    scores = []
    # Walk-forward validation
    for i in range(n_splits):
        # Get the different blocks of data for training and validation
        train_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[:(i+1)*len(group.index)//(n_splits+1),:])
        val_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[(i+1)*len(group.index)//(n_splits+1):(i+2)*len(group.index)//(n_splits+1),:])

        # Drop irrelevant columns to get x,y
        x_train, y_train = split_xy(train_df)
        x_val, y_val = split_xy(val_df)

        model = RandomForestRegressor(max_depth=max_depth, bootstrap=bootstrap, random_state=seed, n_jobs=-1)
        model.fit(x_train, y_train)
        split_score = model.score(x_val, y_val)
        scores.append(split_score)
        
    return scores



def train_nn(model, train_df, shift, epochs,
            chkpt_path, best_path, log_path, patience,
            lr=0.0001, img_path=None):
    '''
    Trains a model with Adam and MSE loss, saving backup checkpoints if training is interrupted.
    Also saves best epoch weights in terms of validation loss and has early stopping.
    Plots the training and validation loss and optionally saves the plot.
    Loads the best model weights and returns the model

    Inputs:
    model is the keras model architecture to train
    train_df is the full train/val dataframe (before rescalingn and splitting into x and y)
    shift is the number of days to shift for time dependent variables
    epochs is the maximum number of epochs to train until
    chkpt_path is the path for saving backups in case training is interrupted
    best_path is the path of the best epoch weights
    log_path is the path that the training logs are stored
    lr is the learning rate of Adam
    patience is the number of epochs befor early stopping kics in
    img_path is the path to save the training/val loss plot. If set to None, will not save
    '''
    # Shift and compute the train/val sets (No walk forward validation due to compute limitations)
    # Create lag features for predictors
    lag_df = add_shift_predictors(train_df, shift)

    # Split into train and validation
    train_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[:9*len(group.index)//10,:])
    val_df = lag_df.groupby('iso_code').apply(lambda group: group.iloc[9*len(group.index)//10:,:])

    # Drop irrelevant columns to get x,y
    x_train, y_train = split_xy(train_df)
    x_val, y_val = split_xy(val_df)

    # Compile and fit the Model
    optim = Adam(learning_rate=lr)
    model.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError())

    early_stopping = EarlyStopping(patience=patience)
    restore_callback = BackupAndRestore(backup_dir=chkpt_path, save_freq=1,
                                        delete_checkpoint = False)
    best_chkpt = ModelCheckpoint(
        best_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        save_freq='epoch'
    )

    logger_callback = CSVLogger(log_path, append=True)

    # Read previous losses if they exist
    if os.path.exists(log_path):
        try:
            previous_training_logs = pd.read_csv(log_path)
            print(previous_training_logs)
            if len(previous_training_logs.index) < epochs:
                history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_val, y_val),
                                    callbacks=[early_stopping, restore_callback, logger_callback, best_chkpt])
        except:
            print('Failed to read previous logs. May not exist or may be an error.')
            history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_val, y_val),
                                    callbacks=[early_stopping, restore_callback, logger_callback, best_chkpt])
    else:
        history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_val, y_val),
                                    callbacks=[early_stopping, restore_callback, logger_callback, best_chkpt])


    # Plot the validation loss and traininig loss across different epochs
    updated_logs = pd.read_csv(log_path)
    plt.plot(updated_logs['epoch'], updated_logs['loss'], label='Training loss')
    plt.plot(updated_logs['epoch'], updated_logs['val_loss'], label='Validation loss')

    # Often the loss in the first few epochs are very large, and it's not worth including
    # We take largest epoch loss *1.1 as the top of the graph
    plt.ylim(top=(max(updated_logs['val_loss'][3:]*1.1 + updated_logs['loss'][3:]*1.1)))
    plt.legend()
    if img_path != None:
        plt.savefig(img_path)
    plt.show()

    # Loads best model weights and evaluates on validation set
    model.load_weights(best_path)
    y_pred = model.predict(x_val)
    scorer = keras.metrics.R2Score()
    scorer.update_state(tf.cast(y_val, tf.float32), y_pred)

    val_score = scorer.result()

    return model, val_score
