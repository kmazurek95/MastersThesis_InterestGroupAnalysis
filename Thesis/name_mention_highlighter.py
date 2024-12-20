import os
import json
import pandas as pd
import re
import uuid
from tqdm import tqdm


# ------------------------------- #
# Utility Functions
# ------------------------------- #

def load_json(file_path):
    """
    Load a JSON file where each line is a JSON object.
    Returns a DataFrame containing the data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return pd.DataFrame([json.loads(line) for line in file])


def save_json(df, file_path):
    """
    Save a DataFrame to a JSON file, writing one record per line.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for record in df.to_dict(orient='records'):
            json.dump(record, file, ensure_ascii=False)
            file.write('\n')


def save_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False, escapechar='\\', sep=',', quotechar='"')


def preprocess_data(df):
    """
    Preprocess the DataFrame:
    - Add UUIDs for each paragraph.
    - Add sequence numbers for tracking.
    - Remove duplicate paragraphs.
    """
    # Add unique IDs and sequence numbers
    df['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['seq_paragraph'] = range(1, len(df) + 1)

    # Remove duplicate rows based on 'org_id' and 'paragraph'
    print(f"Duplicates before removal: {df.duplicated(subset=['org_id', 'paragraph']).sum()}")
    df = df.drop_duplicates(subset=['org_id', 'paragraph'])
    print(f"Duplicates after removal: {df.duplicated(subset=['org_id', 'paragraph']).sum()}")

    return df


def highlight_mentions(df, text_col, word_col):
    """
    Highlight occurrences of words (from 'variation') in a text column.
    Words are wrapped with **** for emphasis.
    """
    def add_highlight(text, word):
        return re.sub(re.escape(word), f"****{word}****", text, flags=re.IGNORECASE)

    tqdm.pandas(desc=f"Highlighting mentions in '{text_col}'")
    return df.progress_apply(lambda row: add_highlight(row[text_col], row[word_col]), axis=1)


def aggregate_overlaps(df):
    """
    Aggregate UUIDs for paragraphs to identify overlapping content.
    Adds 'overlap_ids' and 'overlap_count' to each row.
    """
    aggregated = df.groupby('paragraph')['uuid_paragraph'].apply(lambda x: {
        'overlap_ids': x.unique().tolist(),
        'overlap_count': x.nunique() if x.nunique() > 1 else 0
    }).reset_index()

    # Flatten aggregated data and merge it back into the DataFrame
    aggregated = pd.json_normalize(aggregated['uuid_paragraph'])
    return pd.concat([df, aggregated], axis=1)


def clean_text_before_gpo(df, text_col):
    """
    Clean text by removing all content before 'www.gpo.gov' in the specified column.
    """
    def clean_text(text):
        parts = text.split('www.gpo.gov')
        if len(parts) > 1:
            return parts[-1].split(']')[-1].strip()
        return text

    tqdm.pandas(desc="Cleaning text before 'www.gpo.gov'")
    return df[text_col].progress_apply(clean_text)


# ------------------------------- #
# Main Processing Function
# ------------------------------- #

def process_paragraphs(input_file, output_prefix):
    """
    Process a paragraph dataset:
    - Load JSON data into a DataFrame.
    - Preprocess data (deduplication, UUIDs).
    - Highlight word mentions.
    - Clean unwanted prefixes in text.
    - Aggregate overlaps for analysis.
    - Save results in JSON and CSV formats.
    """
    print("\nStep 1: Loading data...")
    df = load_json(input_file)

    print("\nStep 2: Preprocessing data...")
    df = preprocess_data(df)

    print("\nStep 3: Highlighting word mentions...")
    df['paragraph_highlighted'] = highlight_mentions(df, 'paragraph', 'variation')

    print("\nStep 4: Cleaning text content...")
    df['cleaned_paragraph'] = clean_text_before_gpo(df, 'paragraph')
    df['cleaned_paragraph_highlighted'] = highlight_mentions(df, 'cleaned_paragraph', 'variation')

    print("\nStep 5: Aggregating overlaps...")
    df = aggregate_overlaps(df)

    print("\nStep 6: Saving results...")
    save_json(df, f"{output_prefix}_processed.json")
    save_csv(df, f"{output_prefix}_processed.csv")
    print("Processing complete!\n")


# ------------------------------- #
# Script Execution
# ------------------------------- #

if __name__ == "__main__":
    # Set the base directory for input and output files
    base_path = "C://Users//kaleb//OneDrive//Desktop//DATA//Paragraphs//Mentions by name//"

    # Input files for datasets 114 and 115
    datasets = {
        "114": os.path.join(base_path, "CREC_114_paragraphs_name_raw_uuid_UPDATED_ORGS_4-29-2023.json"),
        "115": os.path.join(base_path, "CREC_115_paragraphs_name_raw_uuid_UPDATED_ORGS_4-29-2023.json"),
    }

    # Process each dataset
    for key, input_file in datasets.items():
        print(f"Processing dataset {key}...")
        output_prefix = os.path.join(base_path, f"PARAGRAPHS_name_{key}")
        process_paragraphs(input_file, output_prefix)
