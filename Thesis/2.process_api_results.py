# Import necessary libraries
import os
import json
import pandas as pd

# Configure pandas to display all columns for debugging purposes
pd.set_option('display.max_columns', None)

# Utility function to read newline-delimited JSON files
def read_json_lines(file_path: str) -> pd.DataFrame:
    """Read a JSON lines file into a DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f.readlines()])

# Save DataFrame as CSV and/or JSON
def save_dataframe(df: pd.DataFrame, file_path: str, save_as_csv=True, save_as_json=True):
    """Save a DataFrame as CSV and/or JSON."""
    base_name, _ = os.path.splitext(file_path)
    if save_as_csv:
        csv_path = f"{base_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
    if save_as_json:
        json_path = f"{base_name}.json"
        df.to_json(json_path, orient='records', force_ascii=False, lines=True)

# Add API keys to specified link columns
def add_api_key_to_links(df: pd.DataFrame, columns: list, api_key: str) -> pd.DataFrame:
    """Add an API key to specified link columns."""
    def add_key(link):
        if isinstance(link, str):
            return f"{link}{'&' if '?' in link else '?'}api_key={api_key}"
        return link

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(add_key)
    return df

# Filter DataFrame to retain only specific columns
def filter_columns(df: pd.DataFrame, columns_to_keep: list) -> pd.DataFrame:
    """Filter a DataFrame to keep only specified columns."""
    return df[columns_to_keep]

# Process a JSON file into multiple outputs
def process_file(input_file: str, output_base_path: str, api_key: str, link_columns: list, column_configs: dict):
    """Process a raw JSON file and create multiple filtered outputs."""
    # Load raw data
    df = read_json_lines(input_file)

    # Add API key to specified link columns
    df = add_api_key_to_links(df, link_columns, api_key)

    # Save raw data with API key added
    save_dataframe(df, os.path.join(output_base_path, "processed_raw"))

    # Filter and save subsets of data
    for name, columns_to_keep in column_configs.items():
        filtered_df = filter_columns(df, columns_to_keep)
        save_dataframe(filtered_df, os.path.join(output_base_path, f"processed_{name}"))

# Main script configuration and execution
if __name__ == "__main__":
    # Set input and output paths
    base_input_path = "C://Users//kaleb//OneDrive//Desktop//DATA//Raw API Calls"
    base_output_path = "C://Users//kaleb//OneDrive//Desktop//OUTPUT DATA"
    api_key = "IgmhiBjvtZhuBDFLlFSYkuGucsP2wKc8it97N4ln"

    # Define column configurations for filtered outputs
    column_configs = {
        "NO_TEXT": [...],  # Metadata without text
        "TEXT": [...],     # Includes parsed text
        "TEXT_ONLY": [...] # Only parsed text columns
    }

    # Link columns for API key addition
    link_columns = ['packageLink', 'relatedLink', 'granulesLink']

    # Files to process with their respective output directories
    files_to_process = [
        ("final_result_114.json", os.path.join(base_output_path, "114")),
        ("final_result_115a.json", os.path.join(base_output_path, "115a")),
        ("final_result_115b.json", os.path.join(base_output_path, "115b"))
    ]

    # Process each file
    for input_file, output_path in files_to_process:
        input_path = os.path.join(base_input_path, input_file)
        os.makedirs(output_path, exist_ok=True)
        process_file(input_path, output_path, api_key, link_columns, column_configs)
