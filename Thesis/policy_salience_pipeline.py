import os
import json
import time
import pandas as pd
from tqdm import tqdm
from pytrends.request import TrendReq
import matplotlib.pyplot as plt

# =========================== Utility Functions ============================

def load_dataframe(file_path, file_type='csv'):
    """
    Load data from CSV or JSON file into a DataFrame.
    Args:
        file_path (str): Path to the file.
        file_type (str): Type of file ('csv' or 'json').
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if file_type == 'csv':
        return pd.read_csv(file_path, encoding='utf-8')
    elif file_type == 'json':
        return pd.read_json(file_path, lines=True, encoding='utf-8')
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'json'.")

def save_dataframe(df, file_path, file_type='csv'):
    """
    Save a DataFrame to a CSV or JSON file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Output file path.
        file_type (str): File format ('csv' or 'json').
    """
    if file_type == 'csv':
        df.to_csv(file_path, index=False)
    elif file_type == 'json':
        df.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'json'.")

def split_list(lst, n):
    """
    Split a list into chunks of size n.
    Args:
        lst (list): List to split.
        n (int): Chunk size.
    Returns:
        list: List of chunks.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_google_trends(pytrends, constant_topic, topics, date):
    """
    Fetch Google Trends data for a set of topics.
    Args:
        pytrends (TrendReq): Pytrends instance.
        constant_topic (str): Base topic for comparison.
        topics (list): List of topics to query.
        date (str): Date range for trends data.
    Returns:
        pd.DataFrame: Google Trends data.
    """
    kw_list = [constant_topic] + topics
    timeframe = f"{date} {date}"  # Single-day range
    pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US')
    return pytrends.interest_over_time()

# =========================== Data Preparation ============================

# File paths
BASE_DIR = "C://Users//kaleb//OneDrive//Desktop//DATA//COMPLETE//"
granule_file = os.path.join(BASE_DIR, "g.graule_meta_data_CREC_114_AND_115.csv")
prominence_file = os.path.join(BASE_DIR, "paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json")

# Load and merge data
print("Loading data...")
df_granule = load_dataframe(granule_file, file_type='csv')
df_prominence = load_dataframe(prominence_file, file_type='json')

print("Merging data...")
merged_df = pd.merge(df_granule, df_prominence, on="granuleId", how="left")
unique_dates = merged_df['dateIssued'].unique()

# =========================== Google Trends Data Collection ============================

# Setup
print("Setting up Google Trends API...")
pytrends = TrendReq(hl='en-US', tz=360)
constant_topic = 'Economy'
other_topics = [
    'Civil Rights', 'Healthcare', 'Agriculture', 'Employment', 'Education Reform',
    'Climate Change', 'Energy', 'Immigration Policy', 'Infrastructure', 'Law Enforcement',
    'Welfare Policy', 'Affordable Housing', 'Trade Policy', 'National Security',
    'Innovation', 'International Trade', 'Foreign Policy', 'Public Administration',
    'National Parks', 'Arts and Culture'
]
topic_groups = split_list(other_topics, 4)  # Split topics into groups of 4 to handle API limits

# Fetch trends data
print("Collecting Google Trends data...")
all_trends_data = []

for date in tqdm(unique_dates[:10], desc="Processing Dates"):  # Limit to 10 dates for testing
    daily_data = []
    for group in topic_groups:
        trends_data = get_google_trends(pytrends, constant_topic, group, date)
        if not trends_data.empty:
            daily_data.append(trends_data)
        time.sleep(20)  # Avoid API rate limits
    
    if daily_data:
        combined_data = pd.concat(daily_data, axis=1).drop(columns=['isPartial'], errors='ignore')
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]  # Remove duplicate columns
        combined_data['dateIssued'] = date
        all_trends_data.append(combined_data)

# Combine all trends data
print("Combining all trends data...")
final_trends_data = pd.concat(all_trends_data).reset_index()
save_dataframe(final_trends_data, "google_trends_data_combined.csv")

# =========================== Data Analysis ============================

print("Analyzing Google Trends data...")
final_trends_data['date'] = pd.to_datetime(final_trends_data['dateIssued'])
final_trends_data['year'] = final_trends_data['date'].dt.year
final_trends_data['congress'] = final_trends_data['year'].apply(lambda y: 114 if y in [2015, 2016] else 115 if y in [2017, 2018] else None)

# Calculate mean salience by congress and year
policy_areas = [col for col in final_trends_data.columns if col not in ['date', 'dateIssued', 'year', 'congress']]
mean_salience = final_trends_data.groupby(['year', 'congress'])[policy_areas].mean().reset_index()

# Melt data for visualization
policy_number_map = {topic: idx * 100 for idx, topic in enumerate(policy_areas, start=1)}
melted_salience = mean_salience.melt(id_vars=['year', 'congress'], var_name='policy_area', value_name='mean_salience')
melted_salience['issue_number'] = melted_salience['policy_area'].map(policy_number_map)

# Save salience data
save_dataframe(melted_salience, "salience_data_final.csv")

# =========================== Visualization ============================

print("Generating visualization...")
plt.figure(figsize=(15, 7))
for col in policy_areas:
    plt.plot(final_trends_data['date'], final_trends_data[col], label=col)

plt.xlabel('Date')
plt.ylabel('Google Trends Interest')
plt.title('Google Trends Interest for Policy Topics')
plt.legend()
plt.show()

print("Script execution complete.")
