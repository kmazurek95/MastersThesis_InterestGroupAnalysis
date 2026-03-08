import os
import gzip
import json
import pandas as pd
from tqdm import tqdm

def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.gz) and returns a Pandas DataFrame.
    """
    print(f"Reading compressed JSON file from: {file_path}")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
            return pd.read_json(gzfile, orient='records', lines=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

def expand_nested_column(df, column_name, nested_key=None):
    """
    Expands a nested column in a DataFrame into a separate DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the nested column.
        column_name (str): The column to expand.
        nested_key (str, optional): Key to extract nested data, if applicable.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    print(f"Expanding nested column: {column_name}")
    expanded_data = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Expanding {column_name}"):
        granule_id = row.get('granuleId', None)
        nested_data = row.get(column_name, None)

        if not nested_data:
            continue
        
        if nested_key and isinstance(nested_data, dict):
            nested_data = nested_data.get(nested_key, [])
        
        if isinstance(nested_data, dict):
            nested_data = [nested_data]  # Convert to a list for consistency

        for item in nested_data:
            expanded_data.append({**item, 'granuleId': granule_id})
    
    return pd.DataFrame(expanded_data)

def generate_bill_links(row):
    """
    Generates bill package IDs and corresponding API links.
    """
    try:
        bill_version = ""  # Placeholder for additional versioning if needed
        bill_id = f"BILLS-{row['congress']}{row['type'].lower()}{row['number']}{bill_version}"
        link = f"https://api.govinfo.gov/packages/{bill_id}/summary?api_key=YOUR_API_KEY_HERE"  # Get your API key from https://api.data.gov/signup/
        return bill_id, link
    except KeyError as e:
        print(f"Missing key while generating bill links: {e}")
        return None, None

def save_dataframe(df, output_dir, file_name, save_csv=True, save_json=True):
    """
    Saves a DataFrame to the specified directory as CSV and/or JSON.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        output_dir (str): Directory to save files.
        file_name (str): Base name for the output files.
        save_csv (bool): Whether to save as a CSV file.
        save_json (bool): Whether to save as a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_csv:
        csv_path = os.path.join(output_dir, f"{file_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved CSV: {csv_path}")

    if save_json:
        json_path = os.path.join(output_dir, f"{file_name}.json")
        df.to_json(json_path, orient='records', lines=True, force_ascii=False)
        print(f"Saved JSON: {json_path}")

def main():
    """
    Main script to process CREC data and expand nested content.
    """
    DATA_PATH = os.environ.get("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    OUTPUT_PATH = os.environ.get("OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    json_file_path = os.path.join(DATA_PATH, "Raw API Calls", "CREC_114_AND_115.json.gz")
    output_dir = OUTPUT_PATH

    df_CREC = read_compressed_json(json_file_path)
    if df_CREC.empty:
        print("No data found. Exiting.")
        return

    print("Expanding nested data columns...")
    df_references_expanded = expand_nested_column(df_CREC, 'references')
    df_committees_expanded = expand_nested_column(df_CREC, 'committees')
    df_members_expanded = expand_nested_column(df_CREC, 'members')
    df_downloads_expanded = expand_nested_column(df_CREC, 'download')

    print("Generating package IDs and links for references...")
    if not df_references_expanded.empty:
        df_references_expanded[['bill_id', 'link']] = df_references_expanded.apply(
            generate_bill_links, axis=1, result_type='expand'
        )

    print("Merging additional link data...")
    df_links_expanded = df_downloads_expanded.merge(
        df_CREC[['granuleId', 'relatedLink', 'detailsLink', 'packageLink']],
        on='granuleId',
        how='left'
    )

    print("Extracting granular metadata...")
    df_granule_metadata = df_CREC[[
        'dateIssued', 'packageId', 'collectionCode', 'title', 'collectionName',
        'granuleClass', 'granuleId', 'bookNumber', 'pagePrefix', 'subGranuleClass',
        'docClass', 'lastModified', 'category', 'granuleDate', 'time', 'rin',
        'legislativeDay'
    ]]

    print("Saving processed data...")
    save_dataframe(df_references_expanded, output_dir, "references_CREC_114_AND_115")
    save_dataframe(df_committees_expanded, output_dir, "committees_CREC_114_AND_115")
    save_dataframe(df_members_expanded, output_dir, "members_CREC_114_AND_115")
    save_dataframe(df_links_expanded, output_dir, "links_CREC_114_AND_115")
    save_dataframe(df_granule_metadata, output_dir, "granule_metadata_CREC_114_AND_115")

    print("Processing completed successfully!")

if __name__ == "__main__":
    main()
