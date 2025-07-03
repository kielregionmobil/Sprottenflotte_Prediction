#!/usr/bin/env python3
# --- Libraries ---
import base64
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


# --- Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()


# --- Configurations ---
# --- Addix ---
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
access_token_cache = {'token': None, 'expires_at': None}
# --- Github ---
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = st.secrets['NAME_REPO']
# --- Data ---
DATA_FILENAME = 'data/data_temp.csv'
STATIONS_FILENAME = 'data/stations.csv'
BASE_URL = "https://apis.kielregion.addix.io/ql/v2/entities/urn:ngsi-ld:BikeHireDockingStation:KielRegion:"

WEATHER_DATA_FILENAME = 'data/weather_data_temp.csv'
WEATHER_STATIONS_FILENAME  = 'data/weather_stations.csv'
WEATHER_URL = "https://apis.kielregion.addix.io/ql/v2/entities/urn:ngsi-ld:WeatherObserved:OWM:"


# --- Functions ---
def read_csv_from_github(filepath, repo, token, branch="main"):
    """
    Fetches a CSV file from a specified GitHub repository and returns it as a pandas DataFrame.

    Parameters:
    - filepath (str): The path to the CSV file within the GitHub repository.
    - repo (str): The GitHub repository in the format 'username/repository'.
    - token (str): GitHub personal access token for authentication.
    - branch (str, optional): The branch of the repository from which to fetch the file. Defaults to "main".

    Returns:
    - df (pandas.DataFrame): DataFrame constructed from the CSV file on GitHub. Returns None if the request fails.
    """
    # Construct the full URL to access the file on GitHub
    url = f'https://api.github.com/repos/{repo}/contents/{filepath}'#?ref={branch}'
    # Set up authentication headers with the provided GitHub token
    headers = {'Authorization': f'token {token}'}

    # Make an HTTP GET request to the GitHub API
    r = requests.get(url, headers=headers)
    # Check if the request was successful
    if r.status_code != 200:
        log.error(f"----- Failed to get Data file from GitHub: {r.content} ------")
        return None

    # Decode the content from the response and load it into a DataFrame
    file_content = r.json()['content']
    decoded_content = base64.b64decode(file_content).decode('utf-8')
    
    # Read the CSV data into a DataFrame
    df = pd.read_csv(StringIO(decoded_content))
    return df


def update_csv_on_github(new_content, filepath, repo, token, branch="main"):
    """
    Updates a CSV file on GitHub by overwriting the existing file with new content.

    Parameters:
    - new_content (str): The new CSV formatted data to upload as string.
    - filepath (str): The path to the CSV file within the GitHub repository.
    - repo (str): The GitHub repository in the format 'username/repository'.
    - token (str): GitHub personal access token for authentication.
    - branch (str, optional): The branch of the repository where the file is located. Defaults to "main".

    Returns:
    - None: A return value is not provided but exceptions and logs indicate failure or success of the operation.
    """
    # Construct the URL to access the specific file in the GitHub repository
    url = f'https://api.github.com/repos/{repo}/contents/{filepath}'#?ref={branch}'
    headers = {'Authorization': f'token {token}'}

    # Initial request to get the current file SHA to enable update
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        log.error(f"Failed to get file info: {r.content}")
        return

    old_content = r.json()
    sha = old_content['sha']

    # Prepare the data for updating: encode the new content to base64
    content_base64 = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')
    payload = {
        "message": f"Update {filepath} file",  # the commit message
        "content": content_base64,             # new file content in base64
        "sha": sha,                            # blob SHA of the file to update
        "branch": branch,                      # branch where the file is located
    }

    # Send a PUT request to update the file on GitHub
    r = requests.put(url, json=payload, headers=headers)
    if r.status_code == 200:
        log.info("----- Data file updated successfully on GitHub -----")
    else:
        log.error(f"----- Failed to update Data file on GitHub: {r.content} ------")


def request_access_token_if_needed():
    """
    Checks if the current access token is valid and returns it; requests and returns a new one if the current token is expired.

    Parameters:
    - None

    Returns:
    - access_token (str): The valid access token to be used for further API requests.
    """
    global access_token_cache
    current_time = time.time()  # Get the current time in seconds since epoch

    # If there is a token and it hasn't expired, use it
    if access_token_cache['token'] and current_time < access_token_cache['expires_at']:
        expiration_time = datetime.fromtimestamp(access_token_cache['expires_at']).strftime('%Y-%m-%d %H:%M:%S')
        log.info(f"---------- Access Token valid until: {expiration_time}")
        return access_token_cache['token']

    # If the token is expired or not present, request a new one
    new_token = request_access_token(CLIENT_SECRET)
    if new_token:
        # Assume the token validity period is 86400 seconds (24 hours); adjust as per your OAuth provider
        token_validity_duration = 86400

        # Update the token cache with the new token and its expiry time
        access_token_cache['token'] = new_token
        access_token_cache['expires_at'] = current_time + token_validity_duration

        expiration_time = datetime.fromtimestamp(access_token_cache['expires_at']).strftime('%Y-%m-%d %H:%M:%S')
        log.info(f"---------- New Access Token valid until: {expiration_time}")

    return new_token


# def request_access_token(USERNAME_EMAIL, PASSWORD, CLIENT_SECRET):
def request_access_token(CLIENT_SECRET):
    """
    Requests an access token using client credentials from an authorization server.

    This function posts client credentials details to the specified token URL of an OAuth authentication server,
    and tries to retrieve an access token. If successful, the access token is returned.

    Parameters:
    - CLIENT_SECRET (str): The secret key associated with the client application.

    Returns:
    - access_token (str or None): The access token if successfully requested; otherwise, None.
    """
    # URL to retrieve the access token from the OAuth provider
    token_url = 'https://accounts.kielregion.addix.io/realms/infoportal/protocol/openid-connect/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # data = {
    #     'grant_type': 'password',
    #     'username': USERNAME_EMAIL, 
    #     'password': PASSWORD,
    #     'client_id': 'quantumleap',
    #     'client_secret': CLIENT_SECRET
    # }
    data = {
        'grant_type': 'client_credentials',
        'client_id': 'prediction_model_sprottenflotte',
        'client_secret': CLIENT_SECRET
    }

    # Performing the HTTP POST request to get the access token
    response = requests.post(token_url, headers=headers, data=data)

    # Check response status code
    if response.status_code == 200:
        token_data = response.json() # Parse JSON response
        access_token = token_data.get('access_token') # Extract access token
        if access_token:
            log.info("---------- Access Token successfully requested")
            return access_token
        else:
            log.info("---------- Access token is not available in the response.")
            return None
    else:
        log.info(f"---------- Error requesting Access Token: {response.status_code}, {response.text}")
        return None


def get_current_dates():
    """
    Calculates and returns the start and end dates for data fetching operations, rounded to the nearest whole hour.

    This function computes the current time adjusted to UTC and rounds it down to the nearest whole hour to make sure
    that hourly averages are calculated for complete hours when making subsequent API requests.

    Parameters:
    - None

    Returns:
    - (datetime, datetime): A tuple containing the start and end datetime objects. The start date is 24 hours before the end date.
    """
    # Calculate the end date as the current time rounded down to the nearest hour
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Setting start date as exactly 24 hours before the end date
    start_date = end_date - timedelta(days=1)

    return start_date, end_date


def fetch_station_data(station_id, from_date, to_date, BASE_URL, ACCESS_TOKEN):
    """
    Retrieves bike availability data for a specified bike hire docking station over a given time period.

    This function performs an HTTP GET request to a specified API URL, making use of query parameters
    for the specific station ID and date range. The received data reflects the hourly average available 
    number of bikes at the specified station.

    Parameters:
    - station_id (str): Unique identifier for the bike hire docking station.
    - from_date (datetime): The start date and time from when the data is to be fetched.
    - to_date (datetime): The end date and time until when the data is to be fetched.
    - BASE_URL (str): The base URL of the API where data about the bike station is hosted.
    - ACCESS_TOKEN (str): The access token required for authenticating the API request.

    Returns:
    - response_data (dict or None): A dictionary containing the fetched data if the request is successful; otherwise, None.
                                    The dictionary includes hourly averaged available bike numbers. If the request fails,
                                    it logs an error with the status code and message and returns None.

    Raises:
    - Exception: If the response from the server is unsuccessful, logs the error status and message.
    """
    # Construct the full API endpoint URL
    url = f"{BASE_URL}{station_id}"
    # Set headers including the authorization token
    headers = {
        'NGSILD-Tenant': 'infoportal',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    # Define the parameters for the API request based on provided dates and aggregation requirements
    params = {
        'type': 'BikeHireDockingStation',
        'fromDate': from_date.isoformat(),
        'toDate': to_date.isoformat(),
        'attrs': 'availableBikeNumber',
        'aggrPeriod': 'hour',
        'aggrMethod': 'avg'
    }
    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json() # Return the parsed JSON data
    else:
        # Log a detailed error message including request details and response status
        log.info(f'---------- No response for station_id: {station_id}\n          from date: {from_date}\n          to date: {to_date}')
        log.info(f"---------- Error: {response.status_code}, {response.text}")
        return None


def fetch_weather_data(station_id, from_date, to_date, WEATHER_URL, ACCESS_TOKEN):
    """
    Retrieves weather data for a specified observation station over a defined time range.

    This function connects to a weather data API using specific credentials and queries weather 
    observations, such as temperature, wind speed, and precipitation, aggregated hourly over the 
    specified date range.

    Parameters:
    - station_id (str): Unique identifier for the weather observation station.
    - from_date (datetime): Starting datetime for the data retrieval.
    - to_date (datetime): Ending datetime for the data retrieval.
    - WEATHER_URL (str): Base URL of the weather data API.
    - ACCESS_TOKEN (str): Token used for API request authentication.

    Returns:
    - response_data (dict or None): If the request is successful, returns a dictionary representing the weather data
                                  aggregated over selected attributes for each hour between the dates specified.
                                  Returns None if the request is unsuccessful, alongside error logs detailing the issue.
    """
    # Construct the full API endpoint URL
    url = f"{WEATHER_URL}{station_id}"
    # Set headers including the authorization token
    headers = {
        'NGSILD-Tenant': 'infoportal',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    # Define the parameters for the API request based on provided dates and aggregation requirements
    params = {
        'type': 'WeatherObserved',
        'fromDate': from_date.isoformat(),
        'toDate': to_date.isoformat(),
        'attrs': 'temperature,windSpeed,precipitation',
        'aggrPeriod': 'hour',
        'aggrMethod': 'avg'
    }
    # Perform the GET request
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json() # Return the parsed JSON data
    else:
        # Log a detailed error message including request details and response status
        log.info(f'---------- No response for weather_station_id: {station_id}\n          from date: {from_date}\n          to date: {to_date}')
        log.info(f"---------- Error: {response.status_code}, {response.text}")
        return None


def create_dataframe_from_api_data(data):
    """
    Converts data received from an API response into a structured pandas DataFrame.

    This function takes JSON-like data containing time indices, entity identifiers, and attributes,
    and transforms it into a pandas DataFrame. This DataFrame is suitable for further analysis or visualization.
    This approach allows data fetched from APIs to be easily manipulated and analyzed in Python using pandas.

    Parameters:
    - data (dict): API data in dictionary format, expected to contain keys 'index', 'entityId', and 'attributes',
                   where 'attributes' is a list of dictionaries with at least a 'name' and 'values'.

    Returns:
    - df (pandas.DataFrame): A DataFrame where the first columns are 'entityId' and 'time_utc', followed by columns
                             for each attribute in 'attributes'. This DataFrame organizes the data with time indices
                             and provides easy access to different measured attributes.

    Raises:
    - ValueError: Raised if the essential keys 'index', 'entityId', or 'attributes' are missing in the 'data' dictionary.
    """
    # Ensure all required keys are present in the data dictionary
    if not all(key in data for key in ['index', 'entityId', 'attributes']):
        raise ValueError("Data missing one of the essential keys: 'index', 'entityId', 'attributes'")

    # Convert the index to a pandas datetime object as it assumes 'index' is in ISO 8601 format
    time_index = pd.to_datetime(data['index'])

    # Extract the entity ID; assumed to be a string directly under 'entityId'
    entity_id = data['entityId']

    # Extract the number after "KielRegion" or "OWM" from the entityId
    match = re.search(r'(?:KielRegion|OWM):(\d+)', entity_id)
    entity_id_number = match.group(1) if match else ''  # Get the number or set to empty if not found

    # Create a dictionary to collect attribute names and their respective values
    attribute_data = {attr['attrName']: attr.get('values', []) for attr in data['attributes']}
    
    # Generate a DataFrame from the collected attribute data
    df = pd.DataFrame(attribute_data)
    
    # Insert 'entityId' and 'time_utc' in the DataFrame
    df['entityId'] = entity_id_number
    df['time_utc'] = time_index
    # Reorder the columns to have 'entityId' first, then 'time', followed by the rest
    column_order = ['entityId', 'time_utc'] + [col for col in df.columns if col not in ['entityId', 'time_utc']]
    df = df[column_order]

    return df


def update_station_data():
    """
    Processes and updates bike station data by retrieving new data from an API and merging it with existing data.

    This function checks for existing data within a specified range (START_DATE to END_DATE) from a local CSV file.
    It identifies gaps in the data for specific stations and dates, fetches the missing data using API calls,
    combines it with the existing data, sorts the combined data, and saves it back to the CSV file.
    Logs are generated throughout the process to track data fetching and updating status.

    Parameters:
    - DATA_FILENAME (str): File path for reading and saving bike station data.
    - STATIONS_FILENAME (str): File path for fetching station-specific configurations or data.
    - START_DATE (datetime): Start date for fetching data.
    - END_DATE (datetime): End date for fetching data.
    - BASE_URL (str): Base URL for the API from which to fetch data.
    - ACCESS_TOKEN (str): Authentication token used for making API requests.

    Returns:
    - None: The function doesn't return a value but outputs logs about the completion of processing or errors encountered.

    Side Effects:
    - Performs read and write operations on a CSV file.
    - Makes HTTP requests to an external API.
    - Modifies global state through potential manipulations in global variables or data structures used across the application.
    """
    # Log the initiation of the data-fetching process.
    log.info('Data-fetching process started')
    start_time = time.time()

    # Retrieve the current valid start and end dates for data fetching.
    START_DATE, END_DATE = get_current_dates()

    # Attempt to load data file from GitHub.
    try:
        # Read the existing data from the GitHub repository
        old_data_temp = read_csv_from_github(DATA_FILENAME, NAME_REPO, GITHUB_TOKEN)

        # Convert the 'time_utc' column to datetime format for time-based operations.
        old_data_temp['time_utc'] = pd.to_datetime(old_data_temp['time_utc'])
        # Filter out (delete) data records that are before the START_DATE.
        old_data_temp = old_data_temp[old_data_temp['time_utc'] >= START_DATE]
        # Remove duplicate records to maintain uniqueness in the dataset.
        old_data_temp = old_data_temp.drop_duplicates().reset_index(drop=True)
    except Exception as e:
        # Log a message indicating the absence of the data file and necessary action.
        log.info(f'---------- No {STATIONS_FILENAME} file exists. Please provide such file with these columns:\nentityId, time_utc, availableBikeNumber')
        log.info(f'---------- Error: {e}')

    # Attempt to load station configuration data from GitHub.
    try:
        # Read the existing data from the GitHub repository    
        stations_data = read_csv_from_github(STATIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)

        # Extract a list of station IDs for which data needs to be fetched.
        STATION_IDS = stations_data['entityId'].tolist()
    except Exception as e:
        # Log error messages if there's an issue in fetching or parsing the stations data.
        log.info(f'---------- No {STATIONS_FILENAME} file exists. Please provide such file with these columns:\nentityId, station_name, maximum_capacity, longitude, latitude, subarea')
        log.info(f'---------- Error: {e}')

    # Get all timestamps (until now) of the timewindow, needed for the model prediction.
    # Subtract one hour from the END_DATE to ensure the request_start_date is not the same as END_DATE.
    # This avoids fetching data for an incomplete hour if the process runs at the top of the hour.
    full_date_range = pd.date_range(start=START_DATE, end=END_DATE - timedelta(hours=1), freq='h') 

    # Initialize an empty list to hold the DataFrames for each station's data.
    dataframes = []

    # Create a boolean mask that is True for rows where the 'entityId' is in the list of station IDs (STATION_IDS).
    mask = old_data_temp['entityId'].isin(STATION_IDS)
    # Apply the mask to filter the data, removing any entries that do not match the specified station IDs.
    old_data_temp = old_data_temp[mask]

    # Request the necessary access token if the existing one is expired or not available.
    ACCESS_TOKEN = request_access_token_if_needed()

    # Iterate through each station ID to process data required for each station.
    for station_id in STATION_IDS:
        # Filter data for a specific station based on 'entityId'.
        station_data = old_data_temp[old_data_temp['entityId'] == station_id]
        # Extract the available dates from the data that's already been fetched for this station.
        available_dates = station_data['time_utc']
        # Determine which dates are missing from the complete date range that should have data.
        missing_dates = full_date_range[~full_date_range.isin(available_dates)]

        # Proceed only if there are missing dates that need data fetching.
        if not missing_dates.empty:
            # The first missing date will be used as the start point for new data fetching.
            request_start_date = missing_dates[0]
            
            # Fetch missing station data from the specified request start date till the END_DATE.
            data = fetch_station_data(station_id, request_start_date, END_DATE, BASE_URL, ACCESS_TOKEN)
            if data:
                # Convert the fetched data to a DataFrame formatted for easier integration.
                df = create_dataframe_from_api_data(data)

                # Identify any hours within the fetched time range that were still not covered.
                fetched_times = pd.to_datetime(df['time_utc'])
                # Create a complete datetime range between the requested start and end dates.
                all_times = pd.date_range(start=request_start_date, end=END_DATE - timedelta(hours=1), freq='h')
                # Find difference which indicates missing times in the fetched data.
                missing_times = all_times.difference(fetched_times)

                # Handle missing times by filling them with NaN entries in the DataFrame.
                if not missing_times.empty:
                    nan_data = pd.DataFrame({
                        'entityId': [station_id] * len(missing_times),
                        'time_utc': missing_times,
                        # Automatically insert a known placeholder that will be replaced by NaN values.
                        'availableBikeNumber': [-42] * len(missing_times)
                    })
                    # Combine the newly fetched data with the missing time data.
                    df = pd.concat([df, nan_data], ignore_index=True).sort_values(by='time_utc')
                    # Replace the placeholder with NaN to signify missing actual measurements.
                    df = df.replace(-42, np.nan)
                
                # Append the prepared DataFrame to the list of DataFrames to be further processed.
                dataframes.append(df)

    # Check if there are any data frames collected in the list.
    if dataframes:
        # Merge all new data frames that contain information from different stations.
        new_data_temp = pd.concat(dataframes)
        # Convert the 'entityId' to a numeric type for consistent processing.
        new_data_temp['entityId'] = new_data_temp['entityId'].astype('int64')
        # Merge the old data frame (previously existing data) with the new, freshly fetched data.
        combined_data_temp = pd.concat([old_data_temp, new_data_temp])
        # Sort the combined data frame by 'entityId' and 'time_utc' for order and easy traceability.
        combined_data_temp = combined_data_temp.sort_values(by=['entityId', 'time_utc'])
        # Reset the index for good data structure and accessibility.
        updated_data_temp = combined_data_temp.reset_index(drop=True)

        # Update the csv-file in the GitHub repository.
        log.info("----- Start updating file on GitHub -----")
        csv_to_github = updated_data_temp.to_csv(index=False)
        update_csv_on_github(csv_to_github, DATA_FILENAME, NAME_REPO, GITHUB_TOKEN)

        # Copy the updated DataFrame for return or further processing.
        data_temp_df = updated_data_temp.copy()

        # Compute the number of new records and unique stations.
        total_new_records = len(new_data_temp)
        unique_stations = new_data_temp['entityId'].nunique()

        # Log the number of new records and the unique stations fetched.
        log.info(f'---------- {total_new_records} new records fetched for {unique_stations} stations.')
        log.info(f'---------- request start date: {request_start_date}')
        log.info(f'---------- Data successfully fetched and saved for all STATION_IDS.')
        message_type = 'success'
        message_text = f'{total_new_records} neue Datenpunkte für {unique_stations} Stationen abgerufen.'
        log.info(f'---------- Time in UTC:\n          Start Date:  {START_DATE}\n          End Date:    {END_DATE}')
    else:
        # Copy the old data frame if no new data has been fetched.
        data_temp_df = old_data_temp.copy()

        # Log that no new data has been processed and existing data is being used.
        log.info('---------- No new data to process, data for every station is available. Existing data used.')
        message_type = 'info'
        message_text = 'Es sind bereits Daten für alle Stationen vorhanden.'

    # Calculate the time taken for the entire data fetching process.
    process_time = time.time() - start_time
    log.info(f'Data-fetching process completed in {round(process_time, 2)} seconds.')

    # Return the updated data frame and messages about the processing status.
    return data_temp_df, message_type, message_text


def update_weather_data():
    """
    Updates and integrates new weather data into an existing dataset stored on GitHub.

    This function retrieves current weather data for specified weather stations between a starting and ending date. 
    It checks for missing data points within the specified date range, fetches missing data, and merges it with the 
    existing data. Then, the updated dataset is uploaded back to GitHub.

    Parameters:
    - WEATHER_DATA_FILENAME (str): The GitHub file path for reading and writing weather data.
    - NAME_REPO (str): The name of the repository where weather data is stored.
    - GITHUB_TOKEN (str): The authentication token for accessing GitHub.
    - WEATHER_STATIONS_FILENAME (str): The file path for metadata about weather stations.
    - WEATHER_URL (str): The base URL for the weather data API.
    - ACCESS_TOKEN (str): The token used for authenticating API requests.

    Returns:
    - tuple: Contains the updated DataFrame, a message type (success/info), and a message text detailing
             the outcome of the update process.
    """
    log.info('Weather-Data-fetching process started')
    start_time = time.time()

    START_DATE, END_DATE = get_current_dates()


    try:
        old_weather = read_csv_from_github(WEATHER_DATA_FILENAME, NAME_REPO, GITHUB_TOKEN)

        old_weather['time_utc'] = pd.to_datetime(old_weather['time_utc'])
        old_weather = old_weather[old_weather['time_utc'] >= START_DATE]
        old_weather = old_weather.drop_duplicates().reset_index(drop=True)
    except Exception as e:
        log.info(f'---------- No {WEATHER_DATA_FILENAME} file exists. Please provide such file with these columns:\nentityId, time_utc, temperature, windSpeed, precipitation')
        log.info(f'---------- Error: {e}')


    try:
        weather_stations_data = read_csv_from_github(WEATHER_STATIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)

        WEATHER_STATION_IDS = weather_stations_data['entityId'].tolist()
    except Exception as e:
        log.info(f'---------- No {WEATHER_STATIONS_FILENAME} file exists. Please provide such file with these columns:\nentityId, station_name, longitude, latitude')
        log.info(f'---------- Error: {e}')

    full_date_range = pd.date_range(start=START_DATE, end=END_DATE - timedelta(hours=1), freq='h') 

    dataframes = []

    mask = old_weather['entityId'].isin(WEATHER_STATION_IDS)
    old_weather = old_weather[mask]

    ACCESS_TOKEN = request_access_token_if_needed()

    for station_id in WEATHER_STATION_IDS:
        station_data = old_weather[old_weather['entityId'] == station_id]
        available_dates = station_data['time_utc']
        missing_dates = full_date_range[~full_date_range.isin(available_dates)]

        if not missing_dates.empty:
            request_start_date = missing_dates[0]

            data = fetch_weather_data(station_id, request_start_date, END_DATE, WEATHER_URL, ACCESS_TOKEN)
            if data:
                df = create_dataframe_from_api_data(data)

                fetched_times = pd.to_datetime(df['time_utc'])
                all_times = pd.date_range(start=request_start_date, end=END_DATE - timedelta(hours=1), freq='h')
                missing_times = all_times.difference(fetched_times)

                if not missing_times.empty:
                    nan_data = pd.DataFrame({
                        'entityId': [station_id] * len(missing_times),
                        'time_utc': missing_times,
                        'temperature': [-42] * len(missing_times),
                        'windSpeed': [-42] * len(missing_times),
                        'precipitation': [-42] * len(missing_times)
                    })
                    df = pd.concat([df, nan_data], ignore_index=True).sort_values(by='time_utc')
                    df = df.replace(-42, np.nan)
                
                dataframes.append(df)

    if dataframes:
        new_weather = pd.concat(dataframes)
        new_weather['entityId'] = new_weather['entityId'].astype('int64')
        combined_weather = pd.concat([old_weather, new_weather])
        combined_weather = combined_weather.sort_values(by=['entityId', 'time_utc'])
        updated_weather = combined_weather.reset_index(drop=True)

        log.info("----- Start updating file on GitHub -----")
        csv_to_github = updated_weather.to_csv(index=False)
        update_csv_on_github(csv_to_github, WEATHER_DATA_FILENAME, NAME_REPO, GITHUB_TOKEN)

        weather_data_df = updated_weather.copy()

        total_new_records = len(new_weather)
        unique_stations = new_weather['entityId'].nunique()

        log.info(f'---------- {total_new_records} new weather records fetched for {unique_stations} stations.')
        log.info(f'---------- request start date: {request_start_date}')
        log.info(f'---------- Data successfully fetched and saved for all WEATHER_STATION_IDS.')
        message_type = 'success'
        message_text = f'{total_new_records} neue Datenpunkte für {unique_stations} Wetterstationen abgerufen.'
        log.info(f'---------- Time in UTC:\n          Start Date:  {START_DATE}\n          End Date:    {END_DATE}')
    else:
        weather_data_df = old_weather.copy()

        log.info('---------- No new data to process, data for every weatherstation is available. Existing data used.')
        message_type = 'info'
        message_text = 'Es sind bereits Daten für alle Wetterstationen vorhanden.'

    process_time = time.time() - start_time
    log.info(f'Weather-Data-fetching process completed in {round(process_time, 2)} seconds.')

    return weather_data_df, message_type, message_text


# --- Easter Egg --->
def update_random_bike_location(stations_df):
    """
    Selects a random subarea from the stations dataframe and assigns new random geographical coordinates.

    This function is primarily intended as an easter egg or demonstration and not for production use.
    It reads a CSV file containing bike station information, randomly picks a subarea if multiple exist,
    and assigns a random latitude and longitude to that subarea. The new location coordinates are not 
    bounded by any geospatial limits related to the subarea and can technically be anywhere in the world.

    Parameters:
    - stations_df (pandas.DataFrame): Dataframe containing information about bike stations,
                                      particularly expecting a 'subarea' column.

    Returns:
    - tuple: Returns a tuple containing the random subarea, new latitude, and new longitude if
             multiple subareas exist. Returns (None, None, None) if only one or no subareas exist.
             (str or None, float or None, float or None)
    
    Note:
    - Ensure the input DataFrame contains a 'subarea' column with valid data.
    """
    stations_df = pd.read_csv(STATIONS_FILENAME)

    if len(stations_df['subarea'].unique()) > 1:
        random_subarea = random.choice(stations_df['subarea'].unique())
        new_lat = random.uniform(-90.0, 90.0)
        new_lon = random.uniform(-180.0, 180.0)

        return random_subarea, new_lat, new_lon
    
    return None, None, None
# <--- Easter Egg ---