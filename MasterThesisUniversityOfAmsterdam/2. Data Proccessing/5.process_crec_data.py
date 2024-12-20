import os
import gzip
import json
import pandas as pd
from tqdm import tqdm

# -------------------------------
# Helper Functions
# -------------------------------

def read_json_lines(file_path):
    """
    Opens and reads a JSON lines file, returning a list of dictionaries.
    """
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def ensure_list_format(data, keys):
    """
    Ensures specified keys in a list of dictionaries are always in list format.
    """
    print("Converting nested columns to lists...")
    for record in data:
        for key in keys:
            record[key] = record.get(key, []) or []
    return data

def expand_column(df, column_name):
    """
    Takes a DataFrame column with nested data and expands it into a new DataFrame.
    """
    print(f"Expanding data from column: {column_name}")
    expanded_data = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Expanding {column_name}"):
        granule_id = row['granuleId']
        nested_data = row[column_name]

        if nested_data:  # Only process rows with data
            nested_df = pd.DataFrame(nested_data)
            nested_df['granuleId'] = granule_id  # Add granule ID for reference
            expanded_data.append(nested_df)

    return pd.concat(expanded_data, ignore_index=True) if expanded_data else pd.DataFrame()

def explode_and_flatten(df, column_name, keys_to_extract):
    """
    Breaks apart nested dictionaries in a column and extracts specific keys into new columns.
    """
    print(f"Flattening nested data in column: {column_name}")
    df_exploded = df.explode(column_name)

    if column_name in df_exploded.columns and keys_to_extract:
        df_exploded[keys_to_extract] = pd.json_normalize(df_exploded[column_name])

    return df_exploded.drop(columns=[column_name], errors='ignore')

def save_data(df, output_dir, file_name, csv=True, json=False, compress=False):
    """
    Saves a DataFrame as a CSV or JSON file in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    if csv:
        file_ext = "csv.gz" if compress else "csv"
        csv_path = os.path.join(output_dir, f"{file_name}.{file_ext}")
        df.to_csv(csv_path, index=False, compression="gzip" if compress else None)
        print(f"Saved CSV: {csv_path}")

    if json:
        file_ext = "json.gz" if compress else "json"
        json_path = os.path.join(output_dir, f"{file_name}.{file_ext}")
        with gzip.open(json_path, 'wt', encoding='utf-8') if compress else open(json_path, 'w', encoding='utf-8') as file:
            df.to_json(file, orient='records', lines=True)
        print(f"Saved JSON: {json_path}")

def combine_files(file_path_1, file_path_2, file_type='csv'):
    """
    Loads and combines two files (CSV or JSON) into a single DataFrame.
    """
    print(f"Combining data from:\n  {file_path_1}\n  {file_path_2}")
    if file_type == 'csv':
        df1 = pd.read_csv(file_path_1)
        df2 = pd.read_csv(file_path_2)
    elif file_type == 'json':
        df1 = pd.read_json(file_path_1, orient='records', lines=True)
        df2 = pd.read_json(file_path_2, orient='records', lines=True)
    else:
        raise ValueError("Invalid file type. Choose 'csv' or 'json'.")

    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    return combined_df

# -------------------------------
# Main Processing Logic
# -------------------------------

def process_congress_data(file_path, output_dir, congress_label):
    """
    Processes CREC data for a specific congress, expanding and saving nested information.
    """
    print(f"\n--- Processing data for Congress {congress_label} ---")

    # Load and prepare the data
    data = read_json_lines(file_path)
    data = ensure_list_format(data, keys=['committees', 'members', 'references'])

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Expand nested columns
    members_df = expand_column(df, 'members')
    committees_df = expand_column(df, 'committees')
    references_df = expand_column(df, 'references')

    # Flatten references if it has nested content
    if not references_df.empty:
        references_df = explode_and_flatten(references_df, 'contents', ['number', 'congress', 'type'])

    # Save processed data
    save_data(members_df, output_dir, f"members_{congress_label}")
    save_data(committees_df, output_dir, f"committees_{congress_label}")
    save_data(references_df, output_dir, f"references_{congress_label}")

    # Process and save links
    print("Processing links...")
    links_df = df[['granuleId']].merge(
        df[['granuleId', 'packageLink', 'detailsLink', 'granulesLink', 'relatedLink']],
        on='granuleId',
        how='left'
    )
    links_df = links_df.join(df['download'].apply(pd.Series))
    save_data(links_df, output_dir, f"links_{congress_label}")

def combine_and_save_files(output_dir, combined_label, *file_paths):
    """
    Combines multiple files into a single DataFrame and saves the output.
    """
    print(f"\n--- Combining files for {combined_label} ---")
    combined_df = combine_files(file_paths[0], file_paths[1])
    save_data(combined_df, output_dir, combined_label)

# -------------------------------
# Main Function
# -------------------------------

def main():
    """
    Main script for processing and saving CREC data for multiple congresses.
    """
    # File paths and directories
    data_dir = "C:/Users/kaleb/OneDrive/Desktop/DATA/Proccessed API Calls/"
    output_dir = "C:/Users/kaleb/OneDrive/Desktop/OUTPUT DATA/"

    # Process data for each congress
    process_congress_data(f"{data_dir}processed_114_RAW.json", output_dir, "114")
    process_congress_data(f"{data_dir}processed_115a_RAW.json", output_dir, "115")

    # Combine and save outputs for both congresses
    combine_and_save_files(output_dir, "members_combined",
                           f"{output_dir}members_114.csv", f"{output_dir}members_115.csv")
    combine_and_save_files(output_dir, "committees_combined",
                           f"{output_dir}committees_114.csv", f"{output_dir}committees_115.csv")
    combine_and_save_files(output_dir, "references_combined",
                           f"{output_dir}references_114.csv", f"{output_dir}references_115.csv")
    combine_and_save_files(output_dir, "links_combined",
                           f"{output_dir}links_114.csv", f"{output_dir}links_115.csv")

    print("\nProcessing completed successfully!")

# -------------------------------
# Script Execution
# -------------------------------

if __name__ == "__main__":
    main()
