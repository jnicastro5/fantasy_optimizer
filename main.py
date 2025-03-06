import streamlit as st
import fetch_data


st.set_page_config(page_title="Daily Fantasy Optimizer", layout="wide")

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

pinnacle_df = fetch_data.fetch_pinnacle_df(PINNACLE_SPORT_IDS)

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
underdog_df = fetch_data.fetch_underdog_df(UNDERDOG_TO_PINNACLE_STAT_MAPPING)
pinnacle_underdog_df = fetch_data.merge_with_pinnacle_df(underdog_df, pinnacle_df)

# Add a checkbox column
pinnacle_underdog_df[""] = False

# Reorder columns to make checkbox column the first column
columns = [""] + [col for col in pinnacle_underdog_df.columns if col != ""]
pinnacle_underdog_df = pinnacle_underdog_df[columns]

# Display the DataFrame with editable checkboxes
pinnacle_underdog_data_editor = st.data_editor(
    pinnacle_underdog_df,
    column_config={"": st.column_config.CheckboxColumn("")},
    hide_index=True,
    key="data_editor",
)

# Get the updated DataFrame with selected rows
selected_rows = pinnacle_underdog_data_editor[pinnacle_underdog_data_editor[""]]

if len(selected_rows) == 2:
    expected_value = selected_rows["hit_percent"].prod() * 2 - (1 - selected_rows["hit_percent"].prod())
elif len(selected_rows) == 3:
    expected_value = selected_rows["hit_percent"].prod() * 5 - (1 - selected_rows["hit_percent"].prod())
elif len(selected_rows) == 4:
    expected_value = selected_rows["hit_percent"].prod() * 9 - (1 - selected_rows["hit_percent"].prod())
elif len(selected_rows) == 5:
    expected_value = selected_rows["hit_percent"].prod() * 19 - (1 - selected_rows["hit_percent"].prod())
else:
    expected_value = 0

st.sidebar.header("Expected Value")
st.sidebar.write(f"Expected Value: {expected_value:.2f}")

# elif tab_selection == "Pinnacle & PrizePicks":
#     prizepicks_df = fetch_prizepicks_df()
#     pinnacle_prizepicks_df = merge_with_pinnacle_df(prizepicks_df, pinnacle_df)
#     if pinnacle_prizepicks_df is not None and not pinnacle_prizepicks_df.empty:
#         st.dataframe(pinnacle_prizepicks_df)  # Display Pinnacle & PrizePicks DataFrame