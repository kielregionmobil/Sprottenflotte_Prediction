#!/usr/bin/env python3
# --- Libraries ---
import os
import logging
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import numpy as np
import streamlit as st

from data import update_csv_on_github, read_csv_from_github


# --- Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()


# --- Configurations ---
# --- Data/Models ---
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'models/model_rf.joblib'
SCALER_FILENAME = 'models/scaler_rf.joblib'
PREDICTIONS_FILENAME = 'data/predictions_rf.csv'
# --- Github ---
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = st.secrets['NAME_REPO']
    

# --- Function ---
def update_predictions(data_df):
    """
    Updates bike availability predictions using a Random Forest model and checks the dataset's recency.

    This function processes input DataFrame to check for the need for new predictions. It compares the latest
    dataset timestamp with the earliest prediction timestamp to determine whether new predictions are necessary.
    If outdated or missing, it proceeds to generate and upload new predictions to a repository.

    Parameters:
    - data_df (pd.DataFrame): The DataFrame containing recent bike availability data.

    Returns:
    - tuple: A tuple containing three elements:
        1. data_temp_predictions (pd.DataFrame or None): DataFrame containing new or existing prediction data, or
           None if an error occurs during the prediction process.
        2. message_type (str): A string indicating the status of the operation ('info', 'success', 'error').
        3. message_text (str): A detailed message describing the outcome of the operation.

    Note:
    - The function logs significant steps and decisions to aid debugging and operational monitoring.
    """
    # Log the start of the prediction process
    log.info('Random Forest prediction process started')

    # make überprüfung, ob predictions needed at this time? otherwise the predictions would be generated every time the application gets refreshed
    try:
        # Prepare the data
        data_temp = data_df.copy()
        data_temp['time_utc'] = pd.to_datetime(data_temp['time_utc'])
        latest_data_time = data_temp['time_utc'].max() # Latest timestamp of data available
    except Exception as e:
        log.info(f'No {DATA_FILENAME} file found.')
        log.info(f'Error: {e}')
    
    # Attempt to load prediction data from GitHub. 
    try:
        # Read the existing prediction data from the GitHub repository
        data_temp_predictions = read_csv_from_github(PREDICTIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)
        data_temp_predictions['prediction_time_utc'] = pd.to_datetime(data_temp_predictions['prediction_time_utc'])
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min() # Earliest timestamp of predictions available
    except Exception as e:
        # Log a message indicating the absence of the data file and necessary action.
        log.info(f'---------- No {PREDICTIONS_FILENAME} file exists. Please provide such file with these columns:\nentityId, prediction_time_utc, prediction_availableBikeNumber')
        log.info(f'---------- Error: {e}')

    # Check if it's necessary to update the prediction data
    if earliest_prediction_time > latest_data_time:
        log.info("---------- No new predictions necessary, predictions are up to date.")
        message_type = 'info'
        message_text = 'Es sind bereits Predictions für alle Stationen vorhanden.'
        log.info('Random Forest prediction process completed')
        return data_temp_predictions, message_type, message_text # Exit the function if no new predictions are required

    # Attempt to load model file.
    try:
        model = joblib.load(MODEL_FILENAME)
    except Exception as e:
        log.info(f'---------- No {MODEL_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    # Attempt to load scaler file.
    try:
        scaler = joblib.load(SCALER_FILENAME)
    except Exception as e:
        log.info(f'---------- No {SCALER_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    # Perform model predictions for each unique station and compile results
    try:
        dataframes = []
        entityId_list = data_temp.entityId.unique()
        for entity in entityId_list:
            data_for_prediction = data_temp[data_temp['entityId'] == entity].copy()

            # prepare data for prediction
            data_for_prediction['Month'] = data_for_prediction['time_utc'].dt.month
            data_for_prediction['Day'] = data_for_prediction['time_utc'].dt.day
            data_for_prediction['Hour'] = data_for_prediction['time_utc'].dt.hour
            data_for_prediction = data_for_prediction[['Month', 'Day', 'Hour', 'availableBikeNumber']]

            # scale the data
            data_for_prediction_scaled = scaler.transform(data_for_prediction)
            data_for_prediction_scaled_flat = data_for_prediction_scaled.flatten().reshape(1, -1)  # Model requires flat input mould

            # Make the predictions
            predictions_scaled = model.predict(data_for_prediction_scaled_flat)

            # Scale predictions back to original scale
            preds = predictions_scaled.flatten()
            feature_index = 3  
            num_features = data_for_prediction.shape[1]
            dummy_matrix = np.zeros((preds.shape[0], num_features))
            dummy_matrix[:, feature_index] = preds

            predictions_original_scale = scaler.inverse_transform(dummy_matrix)[:, feature_index]

            # Create final prediction dataframe
            start_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
            # Generate a list of timestamps for each prediction
            date_list = [start_date + timedelta(hours=i) for i in range(preds.shape[0])]
            
            # Create DataFrame for current entity predictions
            temp_df = pd.DataFrame({
                'entityId': entity,
                'prediction_time_utc': date_list,
                'prediction_availableBikeNumber': predictions_original_scale.tolist()
            })

            # Add the temporary DataFrame to the list
            dataframes.append(temp_df)

        # Merge all temporary DataFrames into one final DataFrame
        data_temp_predictions = pd.concat(dataframes, ignore_index=True)

        # Update the csv-file in the GitHub repo
        log.info("----- Start updating file on GitHub -----")
        csv_to_github = data_temp_predictions.to_csv(index=False)
        update_csv_on_github(csv_to_github, PREDICTIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)
        
        message_type = 'success'
        message_text = 'Es wurden neue Predictions für alle Stationen gemacht.'
        
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()

        log.info('---------- Predictions made successfully and saved for all STATION_IDS.')
        log.info(f'---------- Time in UTC:\n          Earliest Prediction for: {earliest_prediction_time}\n          Latest Data for:         {latest_data_time}')
        log.info('Random Forest prediction process completed')

        return data_temp_predictions, message_type, message_text

    except Exception as e:
        log.info(f'---------- Error: {e}')
        message_type = 'error'
        message_text = 'Fehler beim machen der Predictions.'
        log.info('Random Forest prediction process completed')

        return None, message_type, message_text
