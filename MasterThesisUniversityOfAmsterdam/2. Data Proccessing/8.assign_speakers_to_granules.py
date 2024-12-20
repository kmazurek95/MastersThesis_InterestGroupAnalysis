import os
import gzip
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# -------------------------------
# Utility Functions
# -------------------------------

def read_compressed_json(file_path: str) -> pd.DataFrame:
    """
    Read a compressed JSON file (e.g., .json.gz) and load it into a DataFrame.
    """
    print(f"Reading compressed JSON file: {file_path}")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
            return pd.read_json(gzfile, orient='records', lines=True)
    except Exception as e:
        print(f"Error reading compressed JSON: {e}")
        return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, file_path: str, file_name: str, save_as_csv=True, save_as_json=True):
    """
    Save a DataFrame to a specified directory in CSV and/or JSON format.
    """
    os.makedirs(file_path, exist_ok=True)
    if save_as_csv:
        csv_file_path = os.path.join(file_path, f"{file_name}.csv")
        print(f"Saving DataFrame as CSV: {csv_file_path}")
        df.to_csv(csv_file_path, encoding='utf-8', index=False)
    if save_as_json:
        json_file_path = os.path.join(file_path, f"{file_name}.json")
        print(f"Saving DataFrame as JSON: {json_file_path}")
        df.to_json(json_file_path, orient='records', lines=True, force_ascii=False)


def read_file(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Load a CSV or JSON file into a DataFrame.
    """
    full_path = os.path.join(file_path, file_name)
    print(f"Loading file: {full_path}")
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(full_path, low_memory=False)
        elif file_name.endswith('.json'):
            return pd.read_json(full_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        return pd.DataFrame()


def filter_dataframe(df: pd.DataFrame, mention_index=None, acronym=None, variations_to_exclude=None) -> pd.DataFrame:
    """
    Apply optional filters to a DataFrame based on column values.
    """
    if mention_index is not None:
        df = df[df['mention_index'] == mention_index]
    if acronym is not None:
        df = df[df['acronym'] == acronym]
    if variations_to_exclude is not None:
        df = df[~df['variation'].isin(variations_to_exclude)]
    return df


# -------------------------------
# Data Processing Functions
# -------------------------------

def merge_dataframes(
    file_path: str,
    read_files: list,
    merge_files: list = None,
    merge_on_column: str = None,
    mention_index=None,
    acronym=None,
    variations_to_exclude=None
) -> dict:
    """
    Read multiple DataFrames from files and optionally merge them on a common column.
    """
    print("Reading and merging DataFrames...")
    dfs = {file: filter_dataframe(read_file(file_path, file), mention_index, acronym, variations_to_exclude) for file in read_files}

    if merge_files:
        print(f"Merging files on column: {merge_on_column}")
        merged_df = dfs[merge_files[0]]
        for file in merge_files[1:]:
            merged_df = merged_df.merge(dfs[file], on=merge_on_column, how='left')
        return {"merged": merged_df}
    return dfs


def process_granule_ids(df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Identify granule IDs with a single speaker and extract relevant details.
    """
    print("Processing granule IDs with a single speaker...")
    granule_id_counts = df_members['granuleId'].value_counts()
    single_speaker_ids = granule_id_counts[granule_id_counts == 1].index.tolist()

    df_filtered = df_members[df_members['granuleId'].isin(single_speaker_ids)].copy()
    df_filtered['speaker_provided'] = df_filtered['memberName'].str.split(',').str[0]

    print(f"Number of granule IDs with a single speaker: {len(single_speaker_ids)}")
    return df_filtered[['granuleId', 'authorityId', 'bioGuideId', 'memberName', 'speaker_provided']]


def analyze_merged_data(df_merged: pd.DataFrame, filtered_granule_ids: list):
    """
    Analyze the merged data by checking the proportion of rows with filtered granule IDs.
    """
    print("Analyzing merged data...")
    num_rows_with_filtered_ids = df_merged['granuleId'].isin(filtered_granule_ids).sum()
    percent_filtered_rows = (num_rows_with_filtered_ids / len(df_merged)) * 100

    print(f"{percent_filtered_rows:.2f}% of rows in the merged DataFrame contain filtered granule IDs.")


# -------------------------------
# Main Script
# -------------------------------

if __name__ == "__main__":
    # Paths and filenames
    input_path = "C://Users//kaleb//OneDrive//Desktop//DATA//COMPLETE//"
    output_path = "C:/Users/kaleb/OneDrive/Desktop/OUTPUT DATA/"
    json_file_path = os.path.join(input_path, 'CREC_114_AND_115.json.gz')

    # Files to process
    files = [
        'paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json',
        'g.members_CREC_114_AND_115.csv',
    ]
    common_column = 'granuleId'

    # Step 1: Load and merge data
    print("\nStep 1: Loading and merging data...")
    dfs = merge_dataframes(input_path, files, merge_on_column=common_column)
    df_prominent_mentions = dfs['paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json']
    df_members = dfs['g.members_CREC_114_AND_115.csv']

    # Step 2: Process granule IDs
    print("\nStep 2: Processing granule IDs...")
    df_filtered = process_granule_ids(df_members)
    save_dataframe(df_filtered, output_path, 'speaker_provided', save_as_csv=True, save_as_json=False)

    # Step 3: Read and process compressed JSON data
    print("\nStep 3: Reading compressed JSON data...")
    df_granule_text = read_compressed_json(json_file_path)
    df_granule_grouped = df_granule_text.groupby('granuleId')['parsed_content_text'].apply(
        lambda x: ' '.join(filter(None, x))
    ).reset_index()

    # Step 4: Merge prominent mentions with granule text
    print("\nStep 4: Merging prominent mentions with granule text...")
    df_merged = pd.merge(df_prominent_mentions, df_granule_grouped, on='granuleId', how='left')

    # Step 5: Analyze merged data
    print("\nStep 5: Analyzing merged data...")
    analyze_merged_data(df_merged, df_filtered['granuleId'].unique())

    print("\n--- Process Completed Successfully ---\n")
