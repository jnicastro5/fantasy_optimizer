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
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SCRAPEOPS_API_KEY = st.secrets["prizepicks"]["key"]
RAPID_API_KEY = st.secrets["pinnacle"]["key"]

st.set_page_config(page_title="Daily Fantasy Optimizer", layout="wide")


# Function to send email
def send_email(to_email, subject, body):
    from_email = "hoboheaters@gmail.com"
    from_password = st.secrets["email"]["password"]

    # Set up the SMTP server
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # TLS
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(from_email, from_password)

        # Craft the email message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add the message body
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        st.success(f"Password sent to {to_email}")

    except Exception as e:
        st.error(f"Failed to send email: {e}")


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return str(int(-100 / (decimal_odds - 1)))


def fetch_markets_for_sport(sport_id):
    # API Request
    conn = http.client.HTTPSConnection("pinnacle-odds.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
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


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
def fetch_prizepicks_df():
    scrapeops_url = 'https://proxy.scrapeops.io/v1/'

    prizepicks_url = 'https://api.prizepicks.com/projections?league_id=7&per_page=10000&state_code=AL&single_stat=true&game_mode=prizepools'

    params = {
        'api_key': SCRAPEOPS_API_KEY,
        'url': prizepicks_url
    }

    proxies = {
        "http": f"http://scrapeops:{SCRAPEOPS_API_KEY}@residential-proxy.scrapeops.io:8181",
        "https": f"http://scrapeops:{SCRAPEOPS_API_KEY}@residential-proxy.scrapeops.io:8181"
    }

    response = requests.get(url=scrapeops_url, params=params, proxies=proxies, verify=True).json()

    data = pd.json_normalize(response['data'])
    included = pd.json_normalize(response['included'])
    inc_cop = included[included['type'] == 'new_player'].copy().dropna(axis=1)
    df = pd.merge(
        data,
        inc_cop,
        how='left',
        left_on=['relationships.new_player.data.id', 'relationships.new_player.data.type'],
        right_on=['id', 'type'],
        suffixes=('', '_new_player')
    )

    # List of columns to keep and their new names
    columns = {
        'attributes.description': 'opponent',
        'attributes.stat_type': 'stat',
        'attributes.line_score': 'pp_line',
        'attributes.odds_type': 'odds_type',
        'attributes.display_name': 'full_name',
        'attributes.league': 'sport_id',
        'attributes.team': 'team',
    }

    # Drop all columns except those in columns dictionary
    df.drop(df.columns.difference(columns.keys()), axis=1, inplace=True)

    # Rename columns
    df.rename(columns=columns, inplace=True)

    # Filter for 'standard' odds_type
    df = df[df['odds_type'] == 'standard']
    df.drop(columns=['odds_type'], inplace=True)

    return df


def fetch_underdog_df(stat_mapping):
    underdog_url = 'https://api.underdogfantasy.com/beta/v5/over_under_lines'

    response = requests.get(url=underdog_url).json()

    over_under_lines = pd.json_normalize(response['over_under_lines'])
    options = pd.json_normalize(over_under_lines['options'].explode())
    over_under_lines = over_under_lines.drop(columns=['options']).join(options, rsuffix='_options')

    players = pd.json_normalize(response['players'])
    appearances = pd.json_normalize(response['appearances'])

    df = pd.merge(
        pd.merge(
            players,
            appearances,
            how='inner',
            left_on='id',
            right_on='player_id'
        ),
        over_under_lines,
        how='inner',
        left_on='id_y',
        right_on='over_under.appearance_stat.appearance_id'
    )

    # Filter for choice of 'higher' to avoid 2 rows for each prop
    df = df[df['choice'] == 'higher']

    # Filter for payout_multipliers of 1
    df['payout_multiplier'] = pd.to_numeric(df['payout_multiplier'], errors='coerce')
    df = df[df['payout_multiplier'] == 1]

    # Combine first_name and last_name into one column called 'full_name'
    df['full_name'] = df['first_name'] + ' ' + df['last_name']

    # List of columns to keep and their new names
    columns = {
        'stat_value': 'ud_line',
        'over_under.appearance_stat.display_stat': 'stat',
        'full_name': 'full_name',
        'sport_id': 'sport_id',
    }

    # Drop all columns except those in lines_columns
    df.drop(df.columns.difference(columns.keys()), axis=1, inplace=True)

    # Rename columns
    df.rename(columns=columns, inplace=True)

    # Replace values in the 'stat' column using the mapping dictionary
    df['stat'] = df['stat'].replace(stat_mapping)

    return df


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


# # Basketball - 3
# # Hockey - 4
# # Baseball - 9
# PINNACLE_SPORT_IDS = [3, 4]
#
# # Map underdog_df stat name to pinnacle_df stat name
# UNDERDOG_TO_PINNACLE_STAT_MAPPING = {
#     # Basketball
#     "Pts + Rebs + Asts": "Pts+Rebs+Asts",
#     "3-Pointers Made": "3 Point FG",
#     "Double Doubles": "Double+Double",
#     # Hockey
#     "Shots": "Shots On Goal",
# }
#
# pinnacle_df = fetch_pinnacle_df(PINNACLE_SPORT_IDS)

# prizepicks_df = fetch_prizepicks_df()
# pinnacle_prizepicks_df = merge_with_pinnacle_df(prizepicks_df, pinnacle_df)
# pinnacle_prizepicks_df.to_csv("pinnacle_prizepicks_df.csv", index=False, encoding="utf-8")

# underdog_df = fetch_underdog_df(UNDERDOG_TO_PINNACLE_STAT_MAPPING)
# pinnacle_underdog_df = merge_with_pinnacle_df(underdog_df, pinnacle_df)
# pinnacle_underdog_df.to_csv("pinnacle_underdog_df.csv", index=False, encoding="utf-8")

# Load credentials from YAML
with open("credentials.yml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

if not st.session_state["authentication_status"]:

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

    if st.button("Forgot Password"):
        try:
            forgot_username, email, random_password = authenticator.forgot_password()
            if forgot_username:
                st.write('hi')
                # Send the random password via email
                subject = "Your New Password"
                body = f"Hello {forgot_username},\n\nYour new password is: {random_password}\n\nPlease change it after logging in."
                send_email(email, subject, body)
                st.success('New password sent securely')
        except Exception as e:
            st.error(e)

    if st.button("Sign Up"):
        try:
            new_email, new_username, new_name = authenticator.register_user(password_hint=False)
            if new_email:
                # Update credentials in memory
                config['credentials']['usernames'][new_username] = {
                    'email': new_email,
                    'name': new_name,
                    'password': authenticator.credentials['usernames'][new_username]['password']
                }

                # Save new credentials back to file
                with open("credentials.yml") as file:
                    yaml.dump(config, file, default_flow_style=False)

                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

# Handle authentication status
if st.session_state["authentication_status"]:
    authenticator.logout()

    if st.button("Reset Password"):
        try:
            if authenticator.reset_password(st.session_state['username']):
                st.success('Password modified successfully')
        except Exception as e:
            st.error(e)

    st.write(f'Welcome *{st.session_state["name"]}*')

    st.title("📊 Fantasy Optimizer")

    # Add a select box to allow switching between tabs
    # tab_selection = st.selectbox("Select a Tab", ["Pinnacle & Underdog", "Pinnacle & PrizePicks"])

    # # if tab_selection == "Pinnacle & Underdog":
    # underdog_df = fetch_underdog_df(UNDERDOG_TO_PINNACLE_STAT_MAPPING)
    # pinnacle_underdog_df = merge_with_pinnacle_df(underdog_df, pinnacle_df)
    #
    # # Initialize session state to track selections
    # if "selected_rows" not in st.session_state:
    #     st.session_state.selected_rows = []
    #
    # # Add a checkbox column, pre-filling values based on session state
    # pinnacle_underdog_df[""] = st.session_state.selected_rows
    # display_columns = [""] + [col for col in pinnacle_underdog_df.columns if col != ""]
    # pinnacle_underdog_df = pinnacle_underdog_df[display_columns]
    #
    # # Create an editable data editor
    # pinnacle_underdog_data_editor = st.data_editor(
    #     pinnacle_underdog_df,
    #     column_config={"selected": st.column_config.CheckboxColumn("")},
    #     hide_index=True,
    #     key="data_editor"
    # )
    #
    # # Get selected rows
    # selected_rows = pinnacle_underdog_data_editor[pinnacle_underdog_data_editor[""]]
    #
    # # Update session state with new selections
    # st.session_state.selected_rows = selected_rows
    #
    # # Calculate expected value based on selected rows
    # num_selected = len(selected_rows)
    # if num_selected == 2:
    #     expected_value = selected_rows["hit_percent"].prod() * 2 - (1 - selected_rows["hit_percent"].prod())
    # elif num_selected == 3:
    #     expected_value = selected_rows["hit_percent"].prod() * 5 - (1 - selected_rows["hit_percent"].prod())
    # elif num_selected == 4:
    #     expected_value = selected_rows["hit_percent"].prod() * 9 - (1 - selected_rows["hit_percent"].prod())
    # elif num_selected == 5:
    #     expected_value = selected_rows["hit_percent"].prod() * 19 - (1 - selected_rows["hit_percent"].prod())
    # else:
    #     expected_value = 0
    #
    # # Display Expected Value
    # st.sidebar.header("Expected Value")
    # st.sidebar.write(f"Expected Value: {expected_value:.2f}")

    # elif tab_selection == "Pinnacle & PrizePicks":
    #     prizepicks_df = fetch_prizepicks_df()
    #     pinnacle_prizepicks_df = merge_with_pinnacle_df(prizepicks_df, pinnacle_df)
    #     if pinnacle_prizepicks_df is not None and not pinnacle_prizepicks_df.empty:
    #         st.dataframe(pinnacle_prizepicks_df)  # Display Pinnacle & PrizePicks DataFrame

elif st.session_state["authentication_status"] is False:
    st.error("Incorrect username or password")
# elif st.session_state["authentication_status"] is None:
#     st.warning("Please enter your credentials")