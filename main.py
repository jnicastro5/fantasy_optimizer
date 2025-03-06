import http.client
import requests
import gzip
import json
import pandas as pd
import numpy as np
from io import BytesIO
import concurrent.futures
from scipy.stats import poisson
import streamlit as st


API_KEY = st.secrets["api"]["key"]

st.set_page_config(page_title="Daily Fantasy Optimizer", layout="wide")


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return str(int(-100 / (decimal_odds - 1)))


def fetch_markets_for_sport(sport_id):
    # API Request
    conn = http.client.HTTPSConnection("pinnacle-odds.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': "pinnacle-odds.p.rapidapi.com",
        'Accept-Encoding': 'gzip'  # Ensure server sends gzip response
    }

    conn.request("GET", f"/kit/v1/special-markets?is_have_odds=true&sport_id={sport_id}", headers=headers)

    res = conn.getresponse()
    compressed_data = res.read()  # Read compressed response

    # Decompress the response
    with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
        decompressed_data = f.read()

    # Convert bytes to string and pasrse JSON
    data_str = decompressed_data.decode("utf-8")
    data_json = json.loads(data_str)

    # Extract relevant data
    return data_json.get("specials", [])


def fetch_pinnacle_df(sport_ids):
    all_markets = []

    # Use ThreadPoolExecutor to fetch data in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_markets_for_sport, sport_id): sport_id for sport_id in sport_ids}
        for future in concurrent.futures.as_completed(futures):
            markets = future.result()
            all_markets.extend(markets)

    # Convert the combined data into a DataFrame and process it
    if all_markets:
        df = pd.DataFrame(all_markets)

        # Select only the 'name', 'category', and 'lines' columns
        df = df[['name', 'category', 'lines']]

        # Filter out rows where category is not 'Player Props'
        df = df[df['category'] == 'Player Props']
        df = df.drop(columns=['category'])

        # Extract values for 'over' and 'under' from the 'lines' column
        df['over_decimal'] = df['lines'].apply(lambda x: next(item['price'] for item in x.values() if item['name'] == 'Over'))
        df['under_decimal'] = df['lines'].apply(lambda x: next(item['price'] for item in x.values() if item['name'] == 'Under'))
        df['pinnacle_line'] = df['lines'].apply(lambda x: next(iter(x.values()))['handicap'])

        # Remove the 'lines' column
        df = df.drop(columns=['lines'])

        # Split the 'name' column into two columns based on the '(' character
        df[['name', 'stat']] = df['name'].str.split(' \(', expand=True)

        # Remove the closing parenthesis from the 'stat' column
        df['stat'] = df['stat'].str.rstrip(')')

        # Create new columns for american odds calculated from decimal odds
        df['over_american'] = df['over_decimal'].apply(decimal_to_american)
        df['under_american'] = df['under_decimal'].apply(decimal_to_american)

        # Calculate percentage based on decimal odds
        df['over_percent'] = 1 / df['over_decimal']
        df['under_percent'] = 1 / df['under_decimal']

        # Drop decimal odds columns
        df.drop(columns=['over_decimal'], inplace=True)
        df.drop(columns=['under_decimal'], inplace=True)

        return df
    else:
        print("No market data found in response.")
        return None


def fetch_prizepicks_df():
    # Define the URL and parameters for prizepicks api request
    api_url = 'https://proxy.scrapeops.io/v1/'
    params = {
        'api_key': '9a289b61-44aa-4c18-aa01-f1dc079fe102',
        'url': 'https://api.prizepicks.com/projections?per_page=10000&state_code=AL&single_stat=true&game_mode=prizepools'
    }

    # Make the request
    response = requests.get(url=api_url, params=params)

    # Parse response content as JSON
    data = response.json()

    # Fetch the prizepicks stat names and lines
    lines = data['data']
    lines_df = pd.json_normalize(lines)

    # List of columns to keep from lines_df and their new names
    lines_columns = {
        # 'attributes.description': 'opponent',
        'attributes.stat_type': 'stat',
        'attributes.line_score': 'fantasy_line',
        'attributes.odds_type': 'odds_type',
        'relationships.new_player.data.id': 'player_id'
    }

    # Drop all columns except those in lines_columns
    lines_df.drop(lines_df.columns.difference(lines_columns.keys()), axis=1, inplace=True)

    # Rename columns
    lines_df.rename(columns=lines_columns, inplace=True)

    # Filter for 'standard' odds_type
    lines_df = lines_df[lines_df['odds_type'] == 'standard']
    lines_df.drop(columns=['odds_type'], inplace=True)

    # Fetch the prizepicks player names
    players = data['included']
    players_df = pd.json_normalize(players)

    # List of columns to keep from players_df and their new names
    players_columns = {
        'attributes.display_name': 'name',
        'attributes.league': 'sport_id',
        # 'attributes.team': 'team',
        'attributes.odds_type': 'odds_type',
        'id': 'player_id'
    }

    # Drop all columns except those in players_columns
    players_df.drop(players_df.columns.difference(players_columns.keys()), axis=1, inplace=True)

    # Rename columns
    players_df.rename(columns=players_columns, inplace=True)

    # Filter unwanted sports
    players_df = players_df[players_df['sport_id'].isin(["NBA", "NHL"])]

    # Merge lines_df and players_df using player_id column that is common to both
    final_df = pd.merge(players_df, lines_df, on='player_id', how='inner')
    final_df.drop(columns=['player_id'], inplace=True)

    return final_df


def fetch_underdog_df(stat_mapping):
    underdog_api = 'https://api.underdogfantasy.com/beta/v5/over_under_lines'

    # Make the request
    response = requests.get(url=underdog_api)

    # Parse response content as JSON
    data = response.json()

    # Fetch underdog stat names and lines
    lines = data['over_under_lines']
    lines_df = pd.json_normalize(lines)

    # Explode the list so that each entry gets its own row
    lines_df = lines_df.explode('options')

    # Normalize the 'options' column separately
    options_df = pd.json_normalize(lines_df['options'])

    # Add back to the main DataFrame (reset index first)
    lines_df = lines_df.reset_index(drop=True)
    options_df = options_df.reset_index(drop=True)

    # Concatenate the new normalized columns with the original DataFrame
    lines_df = pd.concat([lines_df, options_df], axis=1)

    # Drop the original 'options' column since it's now expanded
    lines_df.drop(columns=['options'], inplace=True)

    # List of columns to keep from lines_df and their new names
    lines_columns = {
        'stat_value': 'fantasy_line',
        'over_under.appearance_stat.display_stat': 'stat',
        'over_under.appearance_stat.appearance_id': 'appearance_id',
        'payout_multiplier': 'payout_multiplier',
        'choice_display': 'choice',
    }

    # Drop all columns except those in lines_columns
    lines_df.drop(lines_df.columns.difference(lines_columns.keys()), axis=1, inplace=True)

    # Rename columns
    lines_df.rename(columns=lines_columns, inplace=True)

    # Filter for payout_multipliers of 1
    lines_df['payout_multiplier'] = pd.to_numeric(lines_df['payout_multiplier'], errors='coerce')
    lines_df = lines_df[lines_df['payout_multiplier'] == 1]
    lines_df.drop(columns=['payout_multiplier'], inplace=True)

    # Filter for choice of 'Higher'
    lines_df = lines_df[lines_df['choice'] == 'Higher']
    lines_df.drop(columns=['choice'], inplace=True)

    # Convert line from strings to floats
    lines_df['fantasy_line'] = pd.to_numeric(lines_df['fantasy_line'], errors='coerce')

    # Replace underdog_df stats with pinnacle_df stats
    lines_df['stat'] = lines_df['stat'].replace(stat_mapping)

    # Fetch underdog player names and sport
    players = data['players']
    players_df = pd.json_normalize(players)

    # List of columns to keep from players_df and their new names
    players_columns = {
        'id': 'player_id',
        'first_name': 'first_name',
        'last_name': 'last_name',
        'sport_id': 'sport_id',
    }

    # Drop all columns except those in appearances_columns
    players_df.drop(players_df.columns.difference(players_columns.keys()), axis=1, inplace=True)

    # Rename columns
    players_df.rename(columns=players_columns, inplace=True)

    # Combine first_name and last_name into one column called 'full_name'
    players_df['name'] = players_df['first_name'] + ' ' + players_df['last_name']

    # Drop the original 'first_name' and 'last_name' columns if you no longer need them
    players_df.drop(columns=['first_name', 'last_name'], inplace=True)

    # Fetch underdog appearances
    appearances = data['appearances']
    appearances_df = pd.json_normalize(appearances)

    # List of columns to keep from lines_df and their new names
    appearances_columns = {
        'id': 'appearance_id',
        'player_id': 'player_id',
    }

    # Drop all columns except those in appearances_columns
    appearances_df.drop(appearances_df.columns.difference(appearances_columns.keys()), axis=1, inplace=True)

    # Rename columns
    appearances_df.rename(columns=appearances_columns, inplace=True)

    # Merge players_df and appearances_df using player_id column that is common to both
    final_df = pd.merge(players_df, appearances_df, on='player_id', how='inner')
    final_df.drop(columns=['player_id'], inplace=True)

    # Merge underdog_df and lines_df using appearance_id column that is common to both
    final_df = final_df.merge(lines_df, on='appearance_id', how='inner')

    # Drop the 'appearance_id' column
    final_df.drop(columns=['appearance_id'], inplace=True)

    return final_df


def merge_with_pinnacle_df(df1, df2):  # df2 must be pinnacle_df
    # Merge dfs on "name" and "stat" columns
    merged_df = pd.merge(df2, df1, on=["name", "stat"], how="inner")

    # Adjust percentages according to the difference between pinnacle_line and pinnacle_line
    merged_df['over_percent_adj'] = merged_df['pinnacle_line'] / merged_df['fantasy_line'] * merged_df['over_percent']
    merged_df['under_percent_adj'] = merged_df['fantasy_line'] / merged_df['pinnacle_line'] * merged_df['under_percent']

    # Apply only when fantasy_line != pinnacle_line
    mask = merged_df['fantasy_line'] != merged_df['pinnacle_line']

    merged_df.loc[mask, 'over_percent_adj'] = 1 - poisson.cdf(merged_df.loc[mask, 'fantasy_line'] - 0.5,
                                                              merged_df.loc[mask, 'pinnacle_line'])
    merged_df.loc[mask, 'under_percent_adj'] = poisson.cdf(merged_df.loc[mask, 'fantasy_line'] - 0.5,
                                                           merged_df.loc[mask, 'pinnacle_line'])

    merged_df.loc[mask, 'over_percent_novig'] = (merged_df.loc[mask, 'over_percent_adj'] + merged_df.loc[mask, 'over_percent']) / 2
    merged_df.loc[mask, 'under_percent_novig'] = (merged_df.loc[mask, 'under_percent_adj'] + merged_df.loc[mask, 'under_percent']) / 2

    # When mask is false
    merged_df.loc[~mask, 'over_percent_novig'] = merged_df['over_percent'] / (
                merged_df['over_percent'] + merged_df['under_percent'])
    merged_df.loc[~mask, 'under_percent_novig'] = merged_df['under_percent'] / (
                merged_df['over_percent'] + merged_df['under_percent'])

    merged_df.drop(columns=['over_percent', 'under_percent'], inplace=True)

    merged_df.drop(columns=['over_percent_adj', 'under_percent_adj'], inplace=True)

    # Assign the larger percentage as hit_percent
    merged_df['hit_percent'] = merged_df[['over_percent_novig', 'under_percent_novig']].max(axis=1)
    merged_df['hit_percent'] = merged_df['hit_percent'].round(4)

    # Assign "over" or "under" based on which percentage is higher
    merged_df['over_under'] = np.where(merged_df['over_percent_novig'] > merged_df['under_percent_novig'], "over",
                                       "under")

    merged_df.drop(columns=['over_percent_novig', 'under_percent_novig'], inplace=True)

    # Sort based on hit_percent column
    merged_df = merged_df.sort_values(by="hit_percent", ascending=False)

    merged_df = merged_df.reindex(columns=['name', 'sport_id', 'stat', 'fantasy_line', 'pinnacle_line', 'over_american', 'under_american', 'hit_percent', 'over_under'])

    return merged_df


# Basketball - 3
# Hockey - 4
# Baseball - 9
PINNACLE_SPORT_IDS = [3, 4]

# Map underdog_df stat name to pinnacle_df stat name
UNDERDOG_TO_PINNACLE_STAT_MAPPING = {
    # Basketball
    "Pts + Rebs + Asts": "Pts+Rebs+Asts",
    "3-Pointers Made": "3 Point FG",
    "Double Doubles": "Double+Double",
    # Hockey
    "Shots": "Shots On Goal",
}

pinnacle_df = fetch_pinnacle_df(PINNACLE_SPORT_IDS)

# prizepicks_df = fetch_prizepicks_df()
# pinnacle_prizepicks_df = merge_with_pinnacle_df(prizepicks_df, pinnacle_df)
# pinnacle_prizepicks_df.to_csv("pinnacle_prizepicks_df.csv", index=False, encoding="utf-8")

# underdog_df = fetch_underdog_df(UNDERDOG_TO_PINNACLE_STAT_MAPPING)
# pinnacle_underdog_df = merge_with_pinnacle_df(underdog_df, pinnacle_df)
# pinnacle_underdog_df.to_csv("pinnacle_underdog_df.csv", index=False, encoding="utf-8")


st.title("📊 Fantasy Optimizer")

# Add a select box to allow switching between tabs
# tab_selection = st.selectbox("Select a Tab", ["Pinnacle & Underdog", "Pinnacle & PrizePicks"])

# if tab_selection == "Pinnacle & Underdog":
underdog_df = fetch_underdog_df(UNDERDOG_TO_PINNACLE_STAT_MAPPING)
pinnacle_underdog_df = merge_with_pinnacle_df(underdog_df, pinnacle_df)

# Initialize session state to track selections
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

# Create an editable data editor
pinnacle_underdog_data_editor = st.data_editor(
    pinnacle_underdog_df,
    column_config={"selected": st.column_config.CheckboxColumn("Select")},
    hide_index=True,
    key="data_editor"
)

# Get selected rows
selected_rows = pinnacle_underdog_data_editor[pinnacle_underdog_data_editor["selected"]]

# Calculate expected value based on selected rows
num_selected = len(selected_rows)
if num_selected == 2:
    expected_value = selected_rows["hit_percent"].prod() * 2 - (1 - selected_rows["hit_percent"].prod())
elif num_selected == 3:
    expected_value = selected_rows["hit_percent"].prod() * 5 - (1 - selected_rows["hit_percent"].prod())
elif num_selected == 4:
    expected_value = selected_rows["hit_percent"].prod() * 9 - (1 - selected_rows["hit_percent"].prod())
elif num_selected == 5:
    expected_value = selected_rows["hit_percent"].prod() * 19 - (1 - selected_rows["hit_percent"].prod())
else:
    expected_value = 0

# Display Expected Value
st.sidebar.header("Expected Value")
st.sidebar.write(f"Expected Value: {expected_value:.2f}")

# elif tab_selection == "Pinnacle & PrizePicks":
#     prizepicks_df = fetch_prizepicks_df()
#     pinnacle_prizepicks_df = merge_with_pinnacle_df(prizepicks_df, pinnacle_df)
#     if pinnacle_prizepicks_df is not None and not pinnacle_prizepicks_df.empty:
#         st.dataframe(pinnacle_prizepicks_df)  # Display Pinnacle & PrizePicks DataFrame