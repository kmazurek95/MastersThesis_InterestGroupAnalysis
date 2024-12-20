import os
import json
import pandas as pd

# ------------------------------- #
# Utility Functions
# ------------------------------- #

def load_json_lines(file_path):
    """
    Load a line-delimited JSON file into a DataFrame.
    
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        pd.DataFrame: Data loaded into a DataFrame.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return pd.DataFrame([json.loads(line) for line in file])

def save_json_lines(df, file_path):
    """
    Save a DataFrame to a line-delimited JSON file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for record in df.to_dict(orient='records'):
            json.dump(record, file, ensure_ascii=False)
            file.write('\n')

def save_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): Path to the output CSV file.
    """
    df.to_csv(file_path, index=False, escapechar='\\', sep=',', quotechar='"')


def process_data(file_114, file_115, congress_114, congress_115, acronym_flag):
    """
    Process and merge two JSON files by adding congress and acronym flags.
    
    Args:
        file_114 (str): Path to the 114th Congress JSON file.
        file_115 (str): Path to the 115th Congress JSON file.
        congress_114 (int): Congress number for the 114th dataset.
        congress_115 (int): Congress number for the 115th dataset.
        acronym_flag (int): Flag indicating acronym data (0 or 1).
    
    Returns:
        pd.DataFrame: Combined DataFrame with added congress and acronym columns.
    """
    # Load both datasets
    df_114 = load_json_lines(file_114)
    df_115 = load_json_lines(file_115)

    # Add metadata columns
    df_114['congress'] = congress_114
    df_114['acronym'] = acronym_flag
    df_115['congress'] = congress_115
    df_115['acronym'] = acronym_flag

    # Combine datasets
    return pd.concat([df_114, df_115], ignore_index=True)

# ------------------------------- #
# Paths and Input Data
# ------------------------------- #

# Set the base directory for input and output files
base_path = "C:/Users/kaleb/OneDrive/Desktop/DATA/Paragraphs/"

# ------------------------------- #
# Process Name Datasets
# ------------------------------- #

# File paths for NAME data
file_114_name = os.path.join(base_path, "PARAGRAPHS_name_114__UPDATED_4-29-2023.json")
file_115_name = os.path.join(base_path, "PARAGRAPHS_name_115__UPDATED_4-29-2023.json")

# Combine and process NAME data
print("Processing NAME datasets...")
joint_name = process_data(file_114_name, file_115_name, congress_114=114, congress_115=115, acronym_flag=0)

# Save NAME results
save_json_lines(joint_name, os.path.join(base_path, "1A_PARAGRAPHS_name_114_AND_115_UPDATED_4-29-2023.json"))
save_csv(joint_name, os.path.join(base_path, "1A_PARAGRAPHS_name_114_AND_115_UPDATED_4-29-2023.csv"))

# ------------------------------- #
# Process Acronym Datasets
# ------------------------------- #

# File paths for ACRONYM data
file_114_acronym = os.path.join(base_path, "PARAGRAPHS_acronym_EXPANDED_114__UPDATED_4-29-2023.json")
file_115_acronym = os.path.join(base_path, "PARAGRAPHS_acronym_EXPANDED_115__UPDATED_4-29-2023.json")

# Combine and process ACRONYM data
print("Processing ACRONYM datasets...")
joint_acronym = process_data(file_114_acronym, file_115_acronym, congress_114=114, congress_115=115, acronym_flag=1)

# Save ACRONYM results
save_json_lines(joint_acronym, os.path.join(base_path, "2B_PARAGRAPHS_acronym_114_AND_115_EXPANDED_UPDATED_4-29-2023.json"))
save_csv(joint_acronym, os.path.join(base_path, "2B_PARAGRAPHS_acronym_114_AND_115_EXPANDED_UPDATED_4-29-2023.csv"))

# ------------------------------- #
# Combine NAME and ACRONYM Data
# ------------------------------- #

print("Combining NAME and ACRONYM datasets...")
joint_name_acronym = pd.concat([joint_name, joint_acronym], ignore_index=True)

# Save combined results
save_json_lines(joint_name_acronym, os.path.join(base_path, "3A_PARAGRAPHS_name_and_acronym_114_AND_115_UPDATED_4-29-2023.json"))
save_csv(joint_name_acronym, os.path.join(base_path, "3A_PARAGRAPHS_name_and_acronym_114_AND_115_UPDATED_4-29-2023.csv"))

# ------------------------------- #
# Process Expanded Datasets
# ------------------------------- #

print("Processing expanded datasets...")
expanded_file_name = os.path.join(base_path, "1B_PARAGRAPHS_name_114_AND_115_EXPANDED_UPDATED_4-29-2023.json")
expanded_file_acronym = os.path.join(base_path, "2B_PARAGRAPHS_acronym_114_AND_115_EXPANDED_UPDATED_4-29-2023.json")

expanded_name = load_json_lines(expanded_file_name)
expanded_acronym = load_json_lines(expanded_file_acronym)

# Combine expanded datasets
joint_expanded_name_acronym = pd.concat([expanded_name, expanded_acronym], ignore_index=True)

# Save expanded combined results
save_json_lines(joint_expanded_name_acronym, os.path.join(base_path, "3B_PARAGRAPHS_name_AND_acronym_114_AND_115_EXPANDED_UPDATED_4-29-2023.json"))
save_csv(joint_expanded_name_acronym, os.path.join(base_path, "3B_PARAGRAPHS_name_AND_acronym_114_AND_115_EXPANDED_UPDATED_4-29-2023.csv"))

# ------------------------------- #
# Summary and Verification
# ------------------------------- #

# Print column information for verification
print("Columns in combined NAME and ACRONYM dataset:")
print(joint_name_acronym.columns)

print("Columns in expanded NAME and ACRONYM dataset:")
print(joint_expanded_name_acronym.columns)

# Print unique UUID count for ACRONYM dataset
unique_uuids = joint_acronym['uuid_paragraph'].nunique()
print(f"Number of unique UUIDs in ACRONYM dataset: {unique_uuids}")
