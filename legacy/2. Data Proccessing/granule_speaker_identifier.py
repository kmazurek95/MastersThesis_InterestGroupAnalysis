import os
import gzip
import json
import pandas as pd
import re
from tqdm import tqdm
from typing import List, Optional

# ========================= Configuration =========================

# File Paths
DATA_PATH = "/home/kalebmazurek/PIPELINE/Data"  # Base directory for input and output files
OUTPUT_PATH = DATA_PATH                        # Output directory
COMPRESSED_JSON_PATH = '/home/kalebmazurek/PIPELINE/Data/CREC_114_AND_115-Copy1.json.gz'

# ========================= Utility Functions =========================

def read_compressed_json(file_path: str) -> pd.DataFrame:
    """
    Reads a compressed JSON file into a DataFrame.

    Args:
        file_path (str): Path to the compressed JSON file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as gzfile:
        return pd.read_json(gzfile, orient='records', lines=True)


def save_dataframe(df: pd.DataFrame, output_path: str, file_name: str, save_as_csv=True, save_as_json=False):
    """
    Save a DataFrame to CSV or JSON format.

    Args:
        df (pd.DataFrame): Data to save.
        output_path (str): Directory where the file will be saved.
        file_name (str): Output file name (no extension).
        save_as_csv (bool): Whether to save as CSV.
        save_as_json (bool): Whether to save as JSON.
    """
    if save_as_csv:
        csv_file = os.path.join(output_path, f"{file_name}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved CSV: {csv_file}")

    if save_as_json:
        json_file = os.path.join(output_path, f"{file_name}.json")
        df.to_json(json_file, orient='records', lines=True)
        print(f"Saved JSON: {json_file}")


def load_file(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Loads a file into a DataFrame (supports CSV and JSON).

    Args:
        file_path (str): Path to the directory containing the file.
        file_name (str): Name of the file to load.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    full_path = os.path.join(file_path, file_name)
    if file_name.endswith('.csv'):
        return pd.read_csv(full_path)
    elif file_name.endswith('.json'):
        return pd.read_json(full_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_name}")

# ========================= Data Processing Functions =========================

def group_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups text content by 'granuleId' and concatenates 'parsed_content_text'.

    Args:
        df (pd.DataFrame): DataFrame with granule text data.

    Returns:
        pd.DataFrame: Grouped text data.
    """
    return df.groupby('granuleId')['parsed_content_text'].apply(
        lambda x: ' '.join(filter(None, x))
    ).reset_index()


def process_single_speaker_granules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies granules with only one speaker and extracts 'speaker_provided'.

    Args:
        df (pd.DataFrame): DataFrame with member information.

    Returns:
        pd.DataFrame: Filtered DataFrame with 'granuleId' and simplified speaker names.
    """
    single_speaker_ids = df['granuleId'].value_counts()[lambda x: x == 1].index
    filtered_df = df[df['granuleId'].isin(single_speaker_ids)].copy()
    filtered_df['speaker_provided'] = filtered_df['memberName'].str.split(',').str[0]
    return filtered_df[['granuleId', 'speaker_provided']]


def preprocess_text(text: str) -> str:
    """
    Cleans text by removing excessive spaces, line breaks, and non-printable characters.

    Args:
        text (str): Original text to preprocess.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return ''.join(c for c in text if c.isprintable())  # Keep only printable characters


def find_speaker_in_paragraph(paragraph: str, original_text: str) -> Optional[str]:
    """
    Identifies the closest speaker mentioned before a given paragraph in the text.

    Args:
        paragraph (str): Target paragraph to analyze.
        original_text (str): Full original text for searching.

    Returns:
        Optional[str]: Detected speaker name or None.
    """
    search_limit = original_text.find(paragraph)  # Limit the search range
    matches = re.finditer(r'(?:Mr\.|Mrs\.|Ms\.)\s*([A-Z\s]+)\.', original_text[:search_limit])
    return matches[-1].group() if matches else None

# ========================= Main Processing Workflow =========================

def main():
    """Main function to orchestrate the analysis and processing."""
    print("\n=== Loading Data Files ===")
    df_prominent = load_file(DATA_PATH, 'paragraphs_NAME_AND_ACRONYM_114_115_CLASSIFIED.json')
    df_members = load_file(DATA_PATH, 'g.members_CREC_114_AND_115.csv')
    df_committees = load_file(DATA_PATH, 'g.committees_CREC_114_AND_115.csv')
    df_references = load_file(DATA_PATH, 'g.references_CREC_114_AND_115.csv')
    df_text = read_compressed_json(COMPRESSED_JSON_PATH)

    # Step 1: Filter granules with single speakers
    print("\n=== Identifying Single-Speaker Granules ===")
    df_speakers = process_single_speaker_granules(df_members)
    save_dataframe(df_speakers, OUTPUT_PATH, 'single_speaker_filtered')

    # Step 2: Group text content by granuleId
    print("\n=== Grouping Text Content by Granule ID ===")
    df_text_grouped = group_text_data(df_text)

    # Step 3: Merge prominent data with grouped text
    print("\n=== Merging Text Data with Prominent Data ===")
    df_merged = df_prominent.merge(df_text_grouped, on='granuleId', how='left')
    print(f"Number of rows with text: {df_merged['parsed_content_text'].notna().sum()}")

    # Step 4: Preprocess text
    print("\n=== Preprocessing Text Content ===")
    tqdm.pandas(desc="Preprocessing Text")
    df_merged['preprocessed_text'] = df_merged['parsed_content_text'].progress_apply(preprocess_text)

    # Step 5: Identify speakers for each paragraph
    print("\n=== Identifying Speakers in Paragraphs ===")
    speakers_by_paragraph = []
    for _, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Finding Speakers"):
        speaker = find_speaker_in_paragraph(row['p1_original'], row['preprocessed_text'])
        speakers_by_paragraph.append((row['granuleId'], row['uuid_paragraph'], speaker))

    # Save speaker analysis
    result_df = pd.DataFrame(speakers_by_paragraph, columns=['granuleId', 'uuid_paragraph', 'speaker'])
    save_dataframe(result_df, OUTPUT_PATH, 'speakers_by_paragraph')

    # Step 6: Merge with member data for final output
    print("\n=== Merging Speaker Data with Member Information ===")
    unique_members = df_members[['memberName', 'bioGuideId']].drop_duplicates()
    unique_members['last_name'] = unique_members['memberName'].str.split(',').str[0]

    final_result = result_df.merge(unique_members, left_on='speaker', right_on='last_name', how='left')
    save_dataframe(final_result, OUTPUT_PATH, 'final_speaker_identified')

    print("\n=== Processing Complete! ===")


if __name__ == "__main__":
    main()
