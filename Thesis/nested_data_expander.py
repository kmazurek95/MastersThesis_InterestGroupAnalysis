import os
import gzip
import pandas as pd
from tqdm import tqdm
from typing import Optional

# ========================= Configuration =========================

# Define file paths and directories
DATA_PATH = "/home/kalebmazurek/PIPELINE/Data"
OUTPUT_DIR = DATA_PATH
COMPRESSED_JSON_PATH = "/home/kalebmazurek/PIPELINE/Data/CREC_114_AND_115-Copy1.json.gz"

# ========================= Utility Functions =========================

def read_compressed_json(file_path: str) -> pd.DataFrame:
    """
    Reads a compressed JSON file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the compressed JSON file.

    Returns:
        pd.DataFrame: The data loaded into a DataFrame.
    """
    print(f"Reading compressed JSON file: {file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
        return pd.read_json(gzfile, orient='records', lines=True)


def save_dataframe(df: pd.DataFrame, output_dir: str, file_name: str, save_as_csv=True, save_as_json=False):
    """
    Saves a DataFrame to CSV and/or JSON format.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_dir (str): Directory where the file will be saved.
        file_name (str): Name of the file (without extension).
        save_as_csv (bool): Whether to save as a CSV file.
        save_as_json (bool): Whether to save as a JSON file.
    """
    if save_as_csv:
        csv_path = os.path.join(output_dir, f"{file_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Saved CSV: {csv_path}")
    if save_as_json:
        json_path = os.path.join(output_dir, f"{file_name}.json")
        df.to_json(json_path, orient='records', lines=True)
        print(f"✅ Saved JSON: {json_path}")


def expand_nested(df: pd.DataFrame, column_name: str, key: Optional[str] = None) -> pd.DataFrame:
    """
    Expands nested JSON data in a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing nested data.
        column_name (str): The column with nested JSON to expand.
        key (Optional[str]): If nested JSON contains a specific key, extract it.

    Returns:
        pd.DataFrame: Flattened DataFrame with expanded nested data.
    """
    print(f"Expanding nested data in column: '{column_name}'")
    expanded_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Expanding '{column_name}'"):
        granule_id = row['granuleId']
        nested_data = row[column_name]

        if not nested_data:
            continue  # Skip rows with empty nested data

        # Extract data using a specific key if provided
        if key:
            nested_data = nested_data.get(key, [])
        
        nested_df = pd.DataFrame(nested_data)
        nested_df['granuleId'] = granule_id  # Add granuleId for reference
        expanded_data.append(nested_df)

    return pd.concat(expanded_data, ignore_index=True) if expanded_data else pd.DataFrame()

# ========================= Data Processing Functions =========================

def melt_and_expand(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Melts a DataFrame to expand nested data in a specified column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (str): The column containing nested data.

    Returns:
        pd.DataFrame: Flattened and expanded data.
    """
    print(f"Melting and expanding '{column_name}'...")
    melted = pd.melt(df, id_vars='granuleId', value_name=column_name).dropna()
    expanded_data = [
        {**row[column_name], 'granuleId': row['granuleId']}
        for _, row in melted.iterrows()
    ]
    return pd.DataFrame(expanded_data)

# ========================= Main Processing Workflow =========================

def main():
    """
    Main script to process, expand, and save nested Congressional Record data.
    """
    # Step 1: Read the compressed JSON file
    print("\n=== Step 1: Reading Compressed JSON Data ===")
    df_crec = read_compressed_json(COMPRESSED_JSON_PATH)

    # Step 2: Expand nested fields into separate DataFrames
    print("\n=== Step 2: Expanding Nested Fields ===")
    df_references = expand_nested(df_crec, 'references')
    df_committees = expand_nested(df_crec, 'committees')
    df_members = expand_nested(df_crec, 'members')
    df_downloads = expand_nested(df_crec, 'download')

    # Step 3: Expand 'contents' key within references
    print("\nExpanding 'contents' within references...")
    df_references_contents = expand_nested(df_references, column_name=0, key='contents')

    # Step 4: Melt and process committees and members
    df_committees_contents = melt_and_expand(df_committees, 'committee')
    df_members_contents = melt_and_expand(df_members, 'member')

    # Step 5: Merge download links with granule metadata
    print("\n=== Step 3: Merging Download Links ===")
    df_links = df_downloads.merge(
        df_crec[['granuleId', 'relatedLink', 'detailsLink', 'packageLink']],
        on='granuleId',
        how='left'
    )

    # Step 6: Extract relevant metadata columns
    print("\n=== Step 4: Extracting Metadata ===")
    granule_columns = [
        'dateIssued', 'packageId', 'collectionCode', 'title', 'collectionName', 'granuleClass',
        'granuleId', 'bookNumber', 'pagePrefix', 'subGranuleClass', 'docClass',
        'lastModified', 'category', 'granuleDate', 'time', 'rin', 'legislativeDay'
    ]
    df_granule_data = df_crec[granule_columns]

    # Step 7: Add granuleId counts for validation
    print("\nAdding granuleId counts for validation...")
    for df in [df_links, df_members_contents, df_committees_contents, df_references_contents, df_granule_data]:
        df['granuleId_count'] = df.groupby('granuleId')['granuleId'].transform('size')

    # Step 8: Save processed DataFrames to CSV
    print("\n=== Step 5: Saving Processed Data ===")
    save_dataframe(df_links, OUTPUT_DIR, "g.links_CREC_114_AND_115")
    save_dataframe(df_members_contents, OUTPUT_DIR, "g.members_CREC_114_AND_115")
    save_dataframe(df_committees_contents, OUTPUT_DIR, "g.committees_CREC_114_AND_115")
    save_dataframe(df_references_contents, OUTPUT_DIR, "g.references_CREC_114_AND_115")
    save_dataframe(df_granule_data, OUTPUT_DIR, "g.granule_meta_data_CREC_114_AND_115")

    print("\n🎉 Processing Complete! All data saved successfully.")

# ========================= Run Script =========================

if __name__ == "__main__":
    main()
