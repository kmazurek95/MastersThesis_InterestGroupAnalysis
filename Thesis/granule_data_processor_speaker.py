import os
import gzip
import json
import pandas as pd
from typing import List, Optional

# ========================= Configuration =========================

# File Paths
DATA_PATH = "/home/kalebmazurek/PIPELINE/Data"  # Base directory where data is stored
OUTPUT_PATH = DATA_PATH                        # Output directory
COMPRESSED_JSON_PATH = "C://Users//kaleb//OneDrive//Desktop//DATA//COMPLETE//CREC_114_AND_115.json.gz"
VARIATIONS_TO_EXCLUDE = [
    'National Labor Relations Board', 'Continuum of Care', 'Political action committee',
    'Small Business and Entrepreneurship', 'Mentor', "America's Promise", 'Reproductive technology',
    # Add the full list here...
]

# ============================= Utility Functions =============================

def read_compressed_json(file_path: str) -> pd.DataFrame:
    """
    Reads a compressed JSON file into a DataFrame.
    
    Args:
        file_path (str): Path to the compressed JSON file.
        
    Returns:
        pd.DataFrame: Data read from the file.
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
        return pd.read_json(gzfile, orient='records', lines=True)

def save_dataframe(df: pd.DataFrame, file_path: str, file_name: str,
                   save_as_csv: bool = True, save_as_json: bool = False):
    """
    Saves a DataFrame as CSV and/or JSON.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Directory where the file will be saved.
        file_name (str): Name of the output file (without extension).
        save_as_csv (bool): Whether to save as CSV.
        save_as_json (bool): Whether to save as JSON.
    """
    if save_as_csv:
        csv_file_path = os.path.join(file_path, f"{file_name}.csv")
        df.to_csv(csv_file_path, encoding='utf-8', index=False)
        print(f"DataFrame saved as CSV: {csv_file_path}")

    if save_as_json:
        json_file_path = os.path.join(file_path, f"{file_name}.json")
        df.to_json(json_file_path, orient='records', lines=True, force_ascii=False)
        print(f"DataFrame saved as JSON: {json_file_path}")

def load_file(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Loads a CSV or JSON file into a DataFrame.

    Args:
        file_path (str): Directory where the file is located.
        file_name (str): File name to load.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    full_path = os.path.join(file_path, file_name)
    if file_name.endswith('.csv'):
        return pd.read_csv(full_path, low_memory=False)
    elif file_name.endswith('.json'):
        return pd.read_json(full_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file format for {file_name}")

def merge_dataframes(dataframes: List[pd.DataFrame], merge_on: str) -> pd.DataFrame:
    """
    Merges multiple DataFrames on a common column.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames to merge.
        merge_on (str): Column name to merge on.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on=merge_on, how='left')
    return merged_df

# ============================= Processing Functions =============================

def filter_data(df: pd.DataFrame, mention_index: Optional[int] = None,
                 acronym: Optional[str] = None, variations_to_exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filters a DataFrame based on provided parameters.
    
    Args:
        df (pd.DataFrame): DataFrame to filter.
        mention_index (Optional[int]): Filter by mention_index (optional).
        acronym (Optional[str]): Filter by acronym (optional).
        variations_to_exclude (Optional[List[str]]): Filter out variations from the list (optional).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if mention_index is not None:
        df = df[df['mention_index'] == mention_index]
    if acronym is not None:
        df = df[df['acronym'] == acronym]
    if variations_to_exclude is not None:
        df = df[~df['variation'].isin(variations_to_exclude)]
    return df

def process_granule_ids(df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Filters granuleIds with only one speaker and extracts 'speaker_provided'.

    Args:
        df_members (pd.DataFrame): DataFrame with member details.

    Returns:
        pd.DataFrame: Filtered and processed DataFrame with 'speaker_provided' column.
    """
    counts = df_members['granuleId'].value_counts()
    single_speaker_ids = counts[counts == 1].index

    df_filtered = df_members[df_members['granuleId'].isin(single_speaker_ids)].copy()
    df_filtered['speaker_provided'] = df_filtered['memberName'].str.split(',').str[0]
    return df_filtered[['granuleId', 'authorityId', 'bioGuideId', 'memberName', 'speaker_provided']]

def group_text_data(df_text: pd.DataFrame) -> pd.DataFrame:
    """
    Groups text content by granuleId and concatenates parsed_content_text.

    Args:
        df_text (pd.DataFrame): DataFrame containing text data.

    Returns:
        pd.DataFrame: Grouped DataFrame with concatenated text for each granuleId.
    """
    return df_text.groupby('granuleId')['parsed_content_text'].apply(
        lambda x: ' '.join(filter(None, x))).reset_index()

# ============================= Main Execution =============================

def main():
    """Main function to process and save data."""
    
    # Step 1: Load required files
    print("Loading files...")
    files = [
        'paragraphs_NAME_AND_ACRONYM_114_115_CLASSIFIED.json',
        'g.members_CREC_114_AND_115.csv',
        'g.graule_meta_data_CREC_114_AND_115.csv'
    ]
    dataframes = [load_file(DATA_PATH, f) for f in files]
    df_prominent, df_members, df_meta = dataframes

    # Step 2: Process and filter data
    print("Processing granule IDs...")
    df_filtered = process_granule_ids(df_members)
    save_dataframe(df_filtered, OUTPUT_PATH, 'speaker_provided')

    # Step 3: Read and group granule text data
    print("Reading and grouping granule text...")
    df_granule_text = read_compressed_json(COMPRESSED_JSON_PATH)
    df_text_grouped = group_text_data(df_granule_text)

    # Step 4: Merge data from prominent and grouped text data
    print("Merging data...")
    df_merged = df_prominent.merge(df_text_grouped, on='granuleId', how='left')
    print(f"Number of rows with non-null text: {df_merged['parsed_content_text'].notna().sum()}")

    # Step 5: Calculate and print statistics
    filtered_ids = df_filtered['granuleId'].unique()
    percentage_filtered = (df_merged['granuleId'].isin(filtered_ids).sum() / len(df_merged)) * 100
    print(f"{percentage_filtered:.2f}% of rows contain granuleIds found in the filtered data.")

if __name__ == "__main__":
    main()
