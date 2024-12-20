import os
import gzip
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# -------------------------------
# Utility Functions
# -------------------------------

def read_compressed_json(file_path: str) -> pd.DataFrame:
    """
    Read a compressed JSON file (e.g., .json.gz) and load it into a DataFrame.
    """
    print(f"Reading compressed JSON file: {file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
        return pd.read_json(gzfile, orient='records', lines=True)


def save_dataframe(df: pd.DataFrame, file_path: str, file_name: str, save_as_csv=True, save_as_json=True):
    """
    Save a DataFrame to CSV and/or JSON in a specified directory.
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


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text by:
    - Removing non-printable characters.
    - Replacing multiple spaces with a single space.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return ''.join(c for c in text if c.isprintable())  # Remove non-printable characters


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV or JSON file.
    """
    print(f"Loading file: {file_path}")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, low_memory=False)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


# -------------------------------
# Data Processing Functions
# -------------------------------

def merge_dataframes(dfs: dict, merge_files: list, merge_on_column: str) -> pd.DataFrame:
    """
    Merge multiple DataFrames on a specified column.
    """
    print("Merging DataFrames...")
    merged_df = dfs[merge_files[0]]
    for file_name in merge_files[1:]:
        merged_df = merged_df.merge(dfs[file_name], on=merge_on_column, how='left')
    return merged_df


def find_speaker_in_paragraph(paragraph: str, original_text: str, speakers: list) -> str:
    """
    Identify the speaker in a paragraph by searching for their name 
    in the original text using regex matching.
    """
    search_limit = original_text.find(paragraph)
    speaker_pattern = r'(?:Mr\.|Mrs\.|Ms\.)\s*([A-Z\s]+)\.'
    last_match = None
    for match in re.finditer(speaker_pattern, original_text[:search_limit]):
        last_match = match
    return last_match.group(0) if last_match else None


def assign_speakers_to_paragraphs(paragraphs, original_texts, dataset_dict, granule_ids, uuid_paragraphs):
    """
    Assign speakers to paragraphs based on their proximity to the text in the original content.
    """
    print("Assigning speakers to paragraphs...")
    speakers_by_paragraph = []
    for paragraph, original_text, granule_id, uuid_paragraph in tqdm(
        zip(paragraphs, original_texts, granule_ids, uuid_paragraphs),
        total=len(paragraphs),
        desc="Processing Paragraphs"
    ):
        speakers = dataset_dict.get(granule_id, [])
        speaker = find_speaker_in_paragraph(paragraph, original_text, speakers)
        speakers_by_paragraph.append((paragraph, speaker, uuid_paragraph))
    return pd.DataFrame(speakers_by_paragraph, columns=['paragraph', 'speaker', 'uuid_paragraph'])


# -------------------------------
# Main Script
# -------------------------------

if __name__ == "__main__":
    # Define input and output paths
    input_path = "C://Users//kaleb//OneDrive//Desktop//DATA//COMPLETE//"
    output_path = "C://Users//kaleb//OneDrive//Desktop//OUTPUT DATA//"
    json_file_path = os.path.join(input_path, 'CREC_114_AND_115.json.gz')

    # Step 1: Load and preprocess the main dataset
    print("\nStep 1: Loading and preprocessing granule text data...")
    df_granule_text = read_compressed_json(json_file_path)
    df_granule_text['parsed_content_text'] = df_granule_text['parsed_content_text'].apply(preprocess_text)

    # Step 2: Group data by granuleId
    print("\nStep 2: Grouping granule text by granuleId...")
    df_granule_grouped = df_granule_text.groupby('granuleId')['parsed_content_text'].apply(
        lambda x: ' '.join(x.dropna())
    ).reset_index()

    # Step 3: Load paragraphs data and merge with grouped granule data
    print("\nStep 3: Loading paragraph data and merging...")
    df_paragraphs = load_dataframe(os.path.join(input_path, 'paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json'))
    df_merged = pd.merge(df_paragraphs, df_granule_grouped, on='granuleId', how='left')

    # Step 4: Preprocess the merged data
    print("\nStep 4: Preprocessing paragraphs and text...")
    tqdm.pandas(desc="Preprocessing paragraphs")
    df_merged['preprocessed_text'] = df_merged['parsed_content_text'].progress_apply(preprocess_text)
    df_merged['paragraph_processed'] = df_merged['p1_original'].progress_apply(preprocess_text)

    # Step 5: Assign speakers to paragraphs
    print("\nStep 5: Assigning speakers...")
    dataset_dict = defaultdict(list)  # Replace with your actual dataset dictionary
    granule_ids = df_merged['granuleId'].tolist()
    paragraphs = df_merged['paragraph_processed'].tolist()
    original_texts = df_merged['preprocessed_text'].tolist()
    uuid_paragraphs = df_merged['uuid_paragraph'].tolist()

    df_speakers = assign_speakers_to_paragraphs(paragraphs, original_texts, dataset_dict, granule_ids, uuid_paragraphs)

    # Step 6: Save the results
    print("\nStep 6: Saving results...")
    save_dataframe(df_speakers, output_path, 'speakers_by_paragraph', save_as_csv=True, save_as_json=True)

    print("\n--- Process Completed Successfully ---\n")
