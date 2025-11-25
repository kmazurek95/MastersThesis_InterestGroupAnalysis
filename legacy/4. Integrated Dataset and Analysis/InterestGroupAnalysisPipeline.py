import json
import pandas as pd
import re
from tqdm import tqdm
from datasketch import MinHash, MinHashLSHForest
from collections import defaultdict

# Configure pandas to display all columns for better debugging
pd.set_option('display.max_columns', None)

# **Function to Load Data**
def load_dataframe(file_path, file_type='csv'):
    """
    Load a dataset from a CSV or JSON file.

    Parameters:
    - file_path (str): Path to the file.
    - file_type (str): The type of file ('csv' or 'json').

    Returns:
    - pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    if file_type == 'csv':
        return pd.read_csv(file_path, encoding='utf-8')
    elif file_type == 'json':
        return pd.read_json(file_path, orient='records', lines=True)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")

# **Function to Compute Overlapping Variance**
def compute_overlapping_variance(df):
    """
    Analyze overlapping variations and their unique organization IDs.

    Parameters:
    - df (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: DataFrame summarizing variations, org_ids, and counts.
    """
    # Get unique org_ids for each variation
    variations_org_id = df.groupby('variation')['org_id'].unique().reset_index()
    variations_org_id.columns = ['variation', 'org_ids']

    # Count occurrences of each variation
    variations_counts = df.groupby('variation').size().reset_index(name='count')

    # Combine the two summaries
    final_df = pd.merge(variations_org_id, variations_counts, on='variation')
    return final_df

# **Function to Generate MinHash**
def get_minhash(text):
    """
    Create a MinHash object for a given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - MinHash: A MinHash object representing the text.
    """
    m = MinHash(num_perm=128)
    for s in text:
        m.update(s.encode('utf8'))
    return m

# **Function to Detect Potential Duplicates**
def detect_duplicates(df, org_id1, org_id2, threshold=0.99, num_results=10):
    """
    Detect potential duplicate records between two organization IDs.

    Parameters:
    - df (pd.DataFrame): Input data.
    - org_id1 (int): The first organization ID.
    - org_id2 (int): The second organization ID.
    - threshold (float): Similarity threshold to consider duplicates.
    - num_results (int): Number of similar results to return.

    Returns:
    - defaultdict: Dictionary of potential duplicate observations.
    """
    # Filter rows for the two specified org_ids
    filtered_df = df[(df['org_id'] == org_id1) | (df['org_id'] == org_id2)]
    all_minhashes = []
    index_to_info = {}

    # Create MinHash objects for each row
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Generating MinHashes"):
        m = get_minhash(row['p1_original'])
        all_minhashes.append((m, row['variation'], row['uuid_paragraph']))
        index_to_info[len(all_minhashes) - 1] = (row['org_id'], row['uuid_paragraph'])
    
    # Build the LSH Forest for fast similarity search
    forest = MinHashLSHForest(num_perm=128)
    for i, (m, _, _) in enumerate(all_minhashes):
        forest.add(i, m)
    forest.index()

    # Find similar records
    potential_duplicates = defaultdict(list)
    for i, (m, _, _) in enumerate(all_minhashes):
        results = forest.query(m, num_results)
        for j in results:
            if i != j and m.jaccard(all_minhashes[j][0]) > threshold:
                org_id1, uuid1 = index_to_info[i]
                org_id2, uuid2 = index_to_info[j]
                if org_id1 != org_id2:  # Only consider duplicates from different org_ids
                    potential_duplicates[org_id1, uuid1].append((org_id2, uuid2))
    
    return potential_duplicates

# **Function to Filter Observations**
def filter_observations(df, variation_orgId_pairs, variations_to_drop):
    """
    Remove invalid observations based on specific rules.

    Parameters:
    - df (pd.DataFrame): Input data.
    - variation_orgId_pairs (list): List of (variation, org_id) pairs to keep.
    - variations_to_drop (list): List of variations to remove.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    - list: UUIDs of filtered-out rows.
    """
    original_shape = df.shape[0]
    uuids_filtered_out_1, uuids_filtered_out_2 = [], []

    # First filtering pass: Remove mismatched variations and org_ids
    for variation, org_id in variation_orgId_pairs:
        mask = (df['variation'] == variation) & (df['org_id'] != org_id)
        uuids_filtered_out_1.extend(df.loc[mask, 'uuid_paragraph'].tolist())
        df = df[~mask]

    # Second filtering pass: Remove unwanted variations
    mask = df['variation'].isin(variations_to_drop)
    uuids_filtered_out_2.extend(df.loc[mask, 'uuid_paragraph'].tolist())
    df = df[~mask]

    uuids_filtered_out = uuids_filtered_out_1 + uuids_filtered_out_2
    print(f"Filtered out {original_shape - df.shape[0]} rows.")
    return df, uuids_filtered_out

# **Function to Save UUIDs**
def save_uuids(uuids, output_path):
    """
    Save filtered UUIDs to a JSON file.

    Parameters:
    - uuids (list): List of UUIDs to save.
    - output_path (str): File path for saving the UUIDs.
    """
    with open(output_path, 'w') as f:
        json.dump(uuids, f)
    print(f"Filtered UUIDs saved to {output_path}")

# **Main Script**
if __name__ == "__main__":
    # File paths
    classified_path = "C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\Data\\paragraphs_NAME_AND_ACRONYM_114_115_CLASSIFIED.json"
    output_path = "C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\Data\\uuids_filtered_out.json"

    # Load data
    df_classified_mentions = load_dataframe(classified_path, file_type='json')
    print(f"Loaded {df_classified_mentions.shape[0]} rows.")

    # Filtering rules
    variation_orgId_pairs = [
        ("Pharmaceutical Research and Manufacturers of America", "2839"),
        ("National Association for Surface Finishing", "100359"),
        ("National Kidney Foundation", "2532"),
        # Add more rules as needed
    ]
    variations_to_drop = ["AACC", "AAI", "USCCB", "AANP", "AAP", "ABS", "ACC", "AMA"]

    # Apply filters
    df_classified_mentions, uuids_filtered_out = filter_observations(
        df_classified_mentions, variation_orgId_pairs, variations_to_drop
    )

    # Save filtered UUIDs
    save_uuids(uuids_filtered_out, output_path)
    print(f"Final DataFrame shape: {df_classified_mentions.shape}")




import pandas as pd
import numpy as np
import os
import json

# **Function to Load Data**
def read_file(file_path, file_name, mention_index=None, acronym=None, variations_to_exclude=None):
    """
    Load data from a CSV or JSON file, with optional filtering.

    Parameters:
    - file_path (str): Directory containing the file.
    - file_name (str): Name of the file to load.
    - mention_index (int, optional): Filter rows by 'mention_index'. Defaults to None.
    - acronym (str, optional): Filter rows by 'acronym'. Defaults to None.
    - variations_to_exclude (list, optional): Exclude specific variations. Defaults to None.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # Combine path and file name
    file_path_full = os.path.join(file_path, file_name)

    # Load the file based on its type
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path_full, low_memory=False)
    elif file_name.endswith('.json'):
        df = pd.read_json(file_path_full, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_name}. Supported formats are CSV and JSON.")

    # Apply filtering conditions
    if mention_index is not None:
        df = df[df['mention_index'] == mention_index]
    if acronym is not None:
        df = df[df['acronym'] == acronym]
    if variations_to_exclude is not None:
        df = df[~df['variation'].isin(variations_to_exclude)]

    return df

# **Function to Merge Multiple DataFrames**
def merge_dataframes(file_path, files_to_read, files_to_merge, merge_on_column, mention_index=None, acronym=None, variations_to_exclude=None):
    """
    Merge multiple DataFrames on a common column.

    Parameters:
    - file_path (str): Directory containing the files.
    - files_to_read (list): Names of files to read.
    - files_to_merge (list): Names of files to merge.
    - merge_on_column (str): Column to use for merging.
    - mention_index (int, optional): Filter rows by 'mention_index'. Defaults to None.
    - acronym (str, optional): Filter rows by 'acronym'. Defaults to None.
    - variations_to_exclude (list, optional): Exclude specific variations. Defaults to None.

    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    # Load all files into a dictionary of DataFrames
    dfs = {file_name: read_file(file_path, file_name, mention_index, acronym, variations_to_exclude) for file_name in files_to_read}

    # Perform the merging process
    if files_to_merge:
        merged_df = dfs[files_to_merge[0]]
        for file_name in files_to_merge[1:]:
            merged_df = merged_df.merge(dfs[file_name], on=merge_on_column, how='left')
        return merged_df
    return dfs

# **Function to Save DataFrame**
def save_dataframe(df, file_path, file_name, save_as_csv=True, save_as_json=True):
    """
    Save a DataFrame to CSV and/or JSON formats.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - file_path (str): Directory to save the file in.
    - file_name (str): Base name of the output file.
    - save_as_csv (bool): Whether to save as CSV. Defaults to True.
    - save_as_json (bool): Whether to save as JSON. Defaults to True.
    """
    if save_as_csv:
        csv_path = os.path.join(file_path, f"{file_name}.csv")
        df.to_csv(csv_path, encoding='utf-8', index=False)
    if save_as_json:
        json_path = os.path.join(file_path, f"{file_name}.json")
        df.to_json(json_path, orient='records', lines=True, force_ascii=False)

# **Function to Process Data into a Wide Format**
def process_dataframe(df, groupby_column, split_column, prefix):
    """
    Transform a DataFrame into a wide format by grouping and splitting data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - groupby_column (str): Column to group by.
    - split_column (str): Column to expand into multiple columns.
    - prefix (str): Prefix for new columns.

    Returns:
    - pd.DataFrame: DataFrame in wide format.
    """
    # Group and split data
    grouped_df = df.groupby(groupby_column)[split_column].apply(lambda x: '|'.join(x.astype(str))).reset_index()
    df_wide = grouped_df[split_column].str.split('|', expand=True)
    df_wide.insert(0, groupby_column, grouped_df[groupby_column])

    # Rename columns for clarity
    for i in range(1, len(df_wide.columns)):
        df_wide.rename(columns={i: f'{prefix}_{i}'}, inplace=True)
    return df_wide

# **Function to Filter Interest Groups**
def filter_interest_groups(df, conditions):
    """
    Remove interest groups based on specific conditions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - conditions (dict): Filtering criteria as column-value pairs.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    for column, values in conditions.items():
        df = df[~df[column].isin(values)]
    df = df.drop_duplicates(subset=['org_id'])
    return df

# **Function to Validate DataFrame**
def validate_dataframe(df, unique_column):
    """
    Check for duplicates and missing values in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to validate.
    - unique_column (str): Column to check for duplicates.

    Returns:
    - bool: True if the DataFrame is valid, False otherwise.
    """
    duplicates = df[df.duplicated(subset=[unique_column], keep=False)]
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicate rows for '{unique_column}'.")
        return False
    return True

# **Function to Add Congress Column**
def add_congress_column(df, date_column, congress_dates):
    """
    Assign a 'congress' column based on date ranges.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - date_column (str): Column containing date values.
    - congress_dates (dict): Congress periods as {congress: (start_date, end_date)}.

    Returns:
    - pd.DataFrame: Updated DataFrame with a 'congress' column.
    """
    df['congress'] = np.nan
    for congress, (start_date, end_date) in congress_dates.items():
        df.loc[(df[date_column] >= start_date) & (df[date_column] < end_date), 'congress'] = congress
    return df

# **Main Script**
if __name__ == "__main__":
    # File paths and settings
    file_path = "C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\Data\\"
    output_file_name = "df_interest_group_prominence_FINAL"
    files_to_read = [
        'paragraphs_NAME_AND_ACRONYM_114_115_CLASSIFIED.json',
        'g.committees_CREC_114_AND_115.csv',
    ]
    merge_column = 'granuleId'
    congress_dates = {
        '114': (pd.to_datetime('2015-01-03'), pd.to_datetime('2017-01-03')),
        '115': (pd.to_datetime('2017-01-03'), pd.to_datetime('2019-01-03')),
    }

    # Load and merge data
    dfs = merge_dataframes(file_path, files_to_read, None, merge_column)

    # Filter interest groups
    interest_group_conditions = {'CATEGORY': ['Corporation', 'Local government']}
    df_interest_groups = filter_interest_groups(dfs['paragraphs_NAME_AND_ACRONYM_114_115_CLASSIFIED.json'], interest_group_conditions)

    # Add 'congress' column
    df_interest_groups = add_congress_column(df_interest_groups, 'dateIssued', congress_dates)

    # Validate DataFrame
    if validate_dataframe(df_interest_groups, 'uuid_paragraph'):
        print("Validation passed!")

    # Save the final DataFrame
    save_dataframe(df_interest_groups, file_path, output_file_name)
