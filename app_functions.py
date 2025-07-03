#!/usr/bin/env python3
# --- Libraries ---
import pandas as pd
import streamlit as st


# --- Helper Functions ---
def print_message(message_type, message_text):
    """
    Displays a message on a Streamlit application interface using different display styles based on the message type.

    This function takes a message type and the actual message text, then displays them using Streamlit's built-in 
    message displays. Depending on the message type, it appropriately formats the message as info, success, or error.

    Parameters:
    - message_type (str): The type of the message, which determines the formatting style. Valid options are 'info',
                          'success', or 'error'.
    - message_text (str): The text to display in the message.

    Returns:
    - st.delta_generator.DeltaGenerator: The outputted Streamlit element for the message, which is one of
                                         Streamlit's Delta Generators ('st.info', 'st.success', or 'st.error').
                                         Returns None if message_type or message_text are not provided.
    """
    if message_type and message_text:
        if message_type == 'info':
            return st.info(message_text)
        elif message_type == 'success':
            return st.success(message_text)
        elif message_type == 'error':
            return st.error(message_text)


def make_dataframe_of_subarea(selected_option, stations_df):
    """
    Filters and returns a DataFrame for the selected subarea.

    This function checks if a user-selected option matches any entry in the 'subarea' column of 
    the stations data DataFrame. If the selected option is 'Alle' (meaning 'All' in German), it 
    returns a DataFrame containing all stations. Otherwise, it filters the DataFrame to include 
    only those rows where the 'subarea' matches the selected option. The DataFrame is then sorted 
    by a 'Delta' column in descending order.

    Parameters:
    - selected_option (str): The subarea to filter on. If the option is 'Alle', all data is returned.
    - stations_df (pandas.DataFrame): DataFrame containing data about stations, including a 'subarea'
                                      and a 'Delta' column used for sorting.

    Returns:
    - pandas.DataFrame: A DataFrame containing data for the specified subarea, sorted by the absolute
                        value of 'Delta' in descending order, and indexed starting from 1.
    """
    if selected_option == 'Alle':
        subarea_df = stations_df.copy()
    else:
        subarea_df = stations_df[stations_df['subarea'] == selected_option]

    # Sort the dataframe by the absolute values of the 'Delta' column in descending order
    subarea_df = subarea_df.sort_values(
        'Delta',
        ascending=False,  # Sort the 'Delta' column in descending order based on the absolute values
        key=lambda col: abs(col)
    ).reset_index(drop=True)

    subarea_df.index += 1  # Adjust the index to start from 1 instead of 0

    return subarea_df


def get_latest_available_bikes(stations_df):
    """
    Retrieves the most recent available bike count for each station.

    This function sorts a DataFrame of bike stations by 'time_utc' in ascending order, ensuring the latest entry
    for each station is at the bottom. It then groups the data by 'entityId' (each station's unique identifier),
    and extracts the last 'availableBikeNumber' from each group, which corresponds to the most recent count
    of available bikes.

    Parameters:
    - stations_df (pandas.DataFrame): DataFrame containing data on bike stations. It must include
                                      columns 'entityId', 'time_utc', and 'availableBikeNumber'.

    Returns:
    - pandas.Series: A series indexed by 'entityId' with the latest 'availableBikeNumber' for each station.
    """
    # Sort the DataFrame by 'time_utc' to make sure the most recent entries come last
    stations_df_sorted = stations_df.sort_values(by='time_utc', ascending=True)

    # Group by 'entityId' and extract the last entry for 'availableBikeNumber' from each group,
    # which represents the most recent data.
    latest_available_bikes = stations_df_sorted.groupby('entityId')['availableBikeNumber'].last()

    return latest_available_bikes


def add_current_capacity_to_stations_df(stations_df, data_df, color_map):
    """
    Adds current capacity, delta, priority, and color information to the stations DataFrame.

    Parameters:
    - stations_df: DataFrame containing information about stations.
    - data_df: DataFrame containing the latest data used to compute the current capacity.

    Returns:
    - Updated stations_df with added columns: 'current_capacity', 'Delta', 'color_info', and 'color'.
    """
    # Get the latest capacity values from data_df
    latest_available_bikes = get_latest_available_bikes(data_df)

    # Add the current capacity values to the stations_df
    stations_df['current_capacity'] = stations_df['entityId'].map(latest_available_bikes).round()#.astype(int)

    # Calculate the Delta to max_capacity
    stations_df['Delta'] = (stations_df['current_capacity'] - stations_df['maximum_capacity'])#.astype(int)

    # Add a new column to indicate color based on station conditions
    stations_df['color_info'] = stations_df.apply(
        lambda row: 'no data' if pd.isna(row['current_capacity'])
                    else 'überfüllt' if row['current_capacity'] >= 0.8 * row['maximum_capacity']
                    else 'zu leer' if row['current_capacity'] <= 0.2 * row['maximum_capacity'] 
                    else 'okay',
        axis=1
    )

    # Map the colors to a new column
    stations_df['color'] = stations_df['color_info'].map(color_map)

    return stations_df


def add_predictions_to_stations_df(stations_df, predictions_df, color_map_predictions):
    """
    Adds 5 prediction information to each station entry in the stations DataFrame and assigns color codes
    based on predicted changes in bike availability.

    This function merges predicted bike availability data into the station data DataFrame. It also includes 
    a color coding step based on a predefined mapping to visually represent forecasted changes in bike availability.

    Parameters:
    - stations_df (pandas.DataFrame): DataFrame containing bike station data including current capacities.
    - predictions_df (pandas.DataFrame): DataFrame containing predictions for bike availability at various times.
    - color_map_predictions (dict): Dictionary mapping color description to specific color codes.

    Returns:
    - pandas.DataFrame: The updated stations DataFrame with new columns for each prediction time and 
                        an additional column for color-based visualization of the predictions.

    Raises:
    - ValueError: If required columns are missing in either the stations_df or predictions_df.
    """
    # Validate presence of necessary columns in the stations DataFrame
    if 'entityId' not in stations_df.columns:
        raise ValueError("stations_df must contain entityId column")
    # Validate presence of necessary columns in the predictions DataFrame
    if 'entityId' not in predictions_df.columns or 'prediction_time_utc' not in predictions_df.columns or 'prediction_availableBikeNumber' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'entityId', 'prediction_time_utc', and 'prediction_availableBikeNumber' columns")
    
    # Pivot the predictions_df so that each prediction time gets its own column
    predictions_pivot = predictions_df.pivot(index='entityId', columns='prediction_time_utc', values='prediction_availableBikeNumber')
    
    # Round the pivoted prediction values to the nearest integer
    predictions_pivot = predictions_pivot.round(0)

    # Reset the column names of the pivoted DataFrame for clarity
    predictions_pivot.columns = [f'prediction_{i+1}h' for i in range(len(predictions_pivot.columns))]
    
    # Merge the predictions with the stations_df
    stations_df = stations_df.merge(predictions_pivot, how='left', on='entityId')

    # Apply the determine_color function
    stations_df['color_info_predictions'] = stations_df.apply(determine_color, axis=1)
    
    # Map descriptive color information to actual color codes provided in the dictionary
    stations_df['color_predictions'] = stations_df['color_info_predictions'].map(color_map_predictions)
    
    return stations_df


def determine_color(row):
    """
    Determines a color code based on conditions related to bike station capacities now and in the future.

    This function evaluates the current and predicted bike availability at a station and assigns
    a color code string based on specified conditions. The function processes a single row from
    a DataFrame and assumes specific fields are present in that row. The output is intended to
    facilitate visualiation or alert mechanisms by categorically summarizing bike station statuses.

    Parameters:
    - row (pandas.Series): A series containing necessary data of one bike station including 
      'current_capacity', 'prediction_5h' (predicted capacity after 5 hours),
      and 'maximum_capacity'.

    Returns:
    - str: Returns a string representing the color code based on the conditions evaluated.
      It returns 'no data' if there are missing values or none of the specified conditions are met.

    Examples of returned strings include 'zu leer - zu leer', 'überfüllt - okay', etc., with each
    part of the string representing current and future conditions respectively.
    """
    # This only considers the 5th-hour prediction and the current capacity to determine color
    current = row['current_capacity']
    future = row['prediction_5h']

    if pd.isna(current) or pd.isna(future):
        return 'no data'  # Handle missing data
    
    # Define conditions based on capacity thresholds
    condition_current_full = current >= 0.8 * row['maximum_capacity']
    condition_current_empty = current <= 0.2 * row['maximum_capacity']
    condition_current_okay = not (condition_current_full or condition_current_empty)
    
    condition_future_full = future >= 0.8 * row['maximum_capacity']
    condition_future_empty = future <= 0.2 * row['maximum_capacity']
    condition_future_okay = not (condition_future_full or condition_future_empty)
    
    # Nested condition checks to determine the right color based on the mapping logic provided
    if condition_current_empty:
        if condition_future_empty: return 'zu leer - zu leer'
        elif condition_future_okay: return 'zu leer - okay'
        elif condition_future_full: return 'zu leer - überfüllt'
    elif condition_current_full:
        if condition_future_empty: return 'überfüllt - zu leer'
        elif condition_future_okay: return 'überfüllt - okay'
        elif condition_future_full: return 'überfüllt - überfüllt'
    elif condition_current_okay:
        if condition_future_empty: return 'okay - zu leer'
        elif condition_future_okay: return 'okay - okay'
        elif condition_future_full: return 'okay - überfüllt'
    return 'no data'  # default fallback color


def get_full_df_per_station(stations_df, predictions_df, subarea_df):
    """
    Combines historical and predicted bike availability data into a single DataFrame.

    This function merges historical data from the last 24 hours with future predictions for the next 5 hours into a
    cohesive DataFrame. It also enriches this data with additional station details from another DataFrame and adjusts
    timestamps to a specific timezone.

    Parameters:
    - stations_df (pandas.DataFrame): DataFrame containing historical data on bike availability, indexed by `entityId` and `time_utc`.
    - predictions_df (pandas.DataFrame): DataFrame containing future bike availability predictions, indexed by `entityId` and `prediction_time_utc`.
    - subarea_df (pandas.DataFrame): DataFrame containing additional details about each station including 'subarea', 'station_name', and 'maximum_capacity'.

    Returns:
    - pandas.DataFrame: A DataFrame combining both historical and predicted data, sorted chronologically, with additional station details and adjusted for time zone.
    """
    # Convert time columns to datetime objects to ensure proper time series manipulation
    stations_df['time_utc'] = pd.to_datetime(stations_df['time_utc'])
    predictions_df['time_utc'] = pd.to_datetime(predictions_df['prediction_time_utc'])
    # Renaming column for consistency before concatenation
    predictions_df['availableBikeNumber'] = predictions_df['prediction_availableBikeNumber']

    # Concatenate the last 24 hours and the next 5 hours of data into one DataFrame
    full_df = pd.concat([stations_df[['entityId','time_utc','availableBikeNumber']], predictions_df[['entityId','time_utc','availableBikeNumber']]], ignore_index=True)
    
    # Sort values by 'entityId' and 'time_utc' for chronological order per station
    full_df = full_df.sort_values(by=['entityId','time_utc']).reset_index(drop=True)
    
    # Merge additional station details into the main DataFrame
    full_df = full_df.merge(subarea_df[['entityId', 'subarea', 'station_name', 'maximum_capacity']], on='entityId', how='left')
    
    # Adjust time to German timezone by adding 1 hour to UTC
    full_df['deutsche_timezone'] = full_df['time_utc'] + pd.Timedelta(hours=1)

    return full_df


# Berechnet absolute Prio - Muss noch in relative prio umberechnet werden
def measures_prio_of_subarea(stations_df:pd.DataFrame, predictions_df:pd.DataFrame, subareas_df) -> pd.DataFrame:
    """
    Calculates a priority score for each subarea based on station capacities and their distribution over time.

    This function aggregates bike station data and computes a priority score for areas needing attention based 
    on current and historical availability of bikes. It considers several scenarios such as levels of bike 
    availability trending below 20% or above 80% of maximum capacity over specific time frames to assign 
    priority scores, which are then averaged by subarea.

    Parameters:
    - stations_df (pd.DataFrame): DataFrame containing real-time data about stations.
    - predictions_df (pd.DataFrame): DataFrame containing predictions data about bike availability.
    - subareas_df (pd.DataFrame): DataFrame containing additional metadata about the subareas including the maximum capacity.

    Returns:
    - pd.DataFrame: A DataFrame sorted by priority score ('subarea_prio') in descending order for each subarea with an
                    index starting from 1. The columns include 'subarea' and 'subarea_prio'.

    Note:
    - Priority scores are computed based on conditions aiming to identify potential issues like congestion (parks getting full)
      or scarcity (stations getting empty).
    """
    # Retrieve a merged DataFrame combining all related data
    full_df = get_full_df_per_station(stations_df, predictions_df, subareas_df)

    first_iteration = True  # Flag to check if it's the first iteration for DataFrame creation
    # Iterate over each station by their unique entityId
    stations = full_df['entityId'].unique()

    for station in stations:
        prio = 0 # Initialize priority score

        # Retrieve specifics about the subarea and station
        teilbereich = full_df[full_df['entityId'] == station]['subarea'].unique()[0]
        max_capacity = subareas_df[subareas_df['subarea'] == teilbereich]['maximum_capacity'].unique()[0]
        station_data = full_df[full_df['entityId'] == station]
        availableBikes = station_data['availableBikeNumber']

        # Compute scores based on current and historic bike availability
        # Adding scores for scenarios where bike count is critically high
        if availableBikes.iloc[-1] >= (0.8 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() >= (0.8 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() >= (0.8 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() >= (0.8 * max_capacity):  # Mean of last 25 values
            prio += 1
        
        # Adding scores for scenarios where bike count is critically low
        if availableBikes.iloc[-1] <= (0.2 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() <= (0.2 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() <= (0.2 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() <= (0.2 * max_capacity):  # Mean of last 25 values
            prio += 1
    
        # Construct a temporary DataFrame for the station with calculated priority
        temp_df = pd.DataFrame({'subarea': [teilbereich], 'Station': [station], 'Prio': [prio]})

        # Setup for DataFrame concatenation for aggregation
        if first_iteration:
            result_df = temp_df 
            first_iteration = False
        else:
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
    # Group by subarea and compute the mean priority score for each subarea
    result_df = result_df.groupby('subarea')['Prio'].apply(lambda x: x.mean()).reset_index(name='subarea_prio')
    # Sort the results by priority score in descending order
    result_df = result_df.sort_values('subarea_prio', ascending=False).reset_index(drop=True)
    # Adjust index to start from 1 for readability and presentation
    result_df.index += 1

    return result_df
