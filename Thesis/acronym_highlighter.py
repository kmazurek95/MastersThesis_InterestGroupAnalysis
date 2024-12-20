import os
import re
import json
import pandas as pd
import uuid
from tqdm import tqdm


# ------------------------------- #
# Utility Functions
# ------------------------------- #

def load_json_files(file_paths):
    """
    Load data from multiple JSON files into a single DataFrame.
    Each line in the JSON files is assumed to be a valid JSON object.
    """
    all_data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error reading line from {file_path}: {line.strip()} - {e}")
    return pd.DataFrame(all_data)


def save_dataframe(df, json_path, csv_path):
    """
    Save the DataFrame to JSON and CSV formats.
    - JSON: Records stored line by line (ideal for large datasets).
    - CSV: Standard comma-separated values.
    """
    df.to_json(json_path, orient='records', lines=True, force_ascii=False)
    df.to_csv(csv_path, index=False, escapechar='\\', sep=',', quotechar='"')


def preprocess_dataframe(df):
    """
    Preprocess the DataFrame:
    - Add UUIDs to paragraphs for unique identification.
    - Remove duplicate rows based on 'org_id' and 'paragraph'.
    - Aggregate overlapping UUIDs for the same paragraph.
    """
    # Add UUIDs and sequence numbers
    df['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['seq_paragraph'] = list(range(1, len(df) + 1))

    # Remove duplicates
    print(f"Duplicates before removal: {df.duplicated(subset=['org_id', 'paragraph']).sum()}")
    df = df.drop_duplicates(subset=['org_id', 'paragraph'])
    print(f"Duplicates after removal: {df.duplicated(subset=['org_id', 'paragraph']).sum()}")

    # Aggregate overlapping UUIDs
    overlap_info = df.groupby('paragraph').apply(lambda group: pd.Series({
        'overlap_ids': group['uuid_paragraph'].unique().tolist(),
        'overlap_count': group['uuid_paragraph'].nunique()
    })).reset_index()
    
    df = df.merge(overlap_info, on='paragraph', how='left')
    return df


def highlight_mentions(df):
    """
    Highlight occurrences of specific words (variations) in paragraphs.
    Words are wrapped in **** to emphasize them.
    """
    def add_highlighted_text(row, column):
        text = row[column]
        word = row['variation']
        return re.sub(re.escape(word), f"****{word}****", text, flags=re.IGNORECASE)
    
    tqdm.pandas(desc="Highlighting mentions")
    df['paragraph_highlighted'] = df.progress_apply(lambda row: add_highlighted_text(row, 'paragraph'), axis=1)
    df['cleaned_paragraph_highlighted'] = df.progress_apply(lambda row: add_highlighted_text(row, 'cleaned_paragraph'), axis=1)
    return df


def create_duplicate_rows(df):
    """
    Duplicate rows for each word occurrence in the text.
    Each mention gets its own unique identifier and adjusted text.
    """
    def duplicate_mentions(row, column):
        text = row[column]
        word = row['variation']
        mentions = re.finditer(re.escape(word), text, flags=re.IGNORECASE)
        duplicates = []
        for idx, match in enumerate(mentions, start=1):
            new_row = row.copy()
            new_row.update({
                f"{column}_mention": text[:match.start()] + f"****({idx}){word}****" + text[match.end():],
                'mention_index': idx,
                'uuid_mention': f"{row['uuid_paragraph']}-{idx}"
            })
            duplicates.append(new_row)
        return duplicates

    all_rows = []
    tqdm.pandas(desc="Creating duplicate rows")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        all_rows.extend(duplicate_mentions(row, 'paragraph'))
        all_rows.extend(duplicate_mentions(row, 'cleaned_paragraph'))
    return pd.DataFrame(all_rows)


def highlight_other_mentions(df):
    """
    Highlight mentions that don't match the main mention index.
    Other mentions are wrapped in ***** for distinction.
    """
    def highlight_others(row, text_col):
        text = row[text_col]
        word = row['variation']
        mentions = list(re.finditer(re.escape(word), text, flags=re.IGNORECASE))
        main_index = row['mention_index']

        # Replace non-main mentions
        for idx, match in reversed(list(enumerate(mentions, start=1))):
            if idx != main_index:
                text = text[:match.start()] + f"*****{word}*****" + text[match.end():]
        return text

    tqdm.pandas(desc="Highlighting other mentions")
    df['cleaned_mention'] = df.progress_apply(lambda row: highlight_others(row, 'cleaned_paragraph_mention'), axis=1)
    return df


# ------------------------------- #
# Main Processing Function
# ------------------------------- #

def process_acronym_data(files, output_prefix):
    """
    Main function to process acronym data:
    - Load JSON files into a DataFrame.
    - Preprocess data: deduplicate, clean, and aggregate.
    - Highlight word mentions and create duplicate rows for analysis.
    - Save the processed data in JSON and CSV formats.
    """
    # Step 1: Load data
    print("\nLoading data...")
    df = load_json_files(files)

    # Step 2: Preprocess and clean data
    print("\nPreprocessing data...")
    df = preprocess_dataframe(df)

    # Step 3: Highlight word mentions
    print("\nHighlighting mentions...")
    df = highlight_mentions(df)

    # Step 4: Clean paragraphs by removing unwanted prefixes
    print("\nCleaning paragraphs...")
    tqdm.pandas(desc="Removing prefixes")
    df['cleaned_paragraph'] = df['paragraph'].progress_apply(
        lambda paragraph: paragraph.split('www.gpo.gov')[-1].split(']')[-1].strip()
    )

    # Step 5: Create duplicate rows for each mention
    print("\nDuplicating rows for mentions...")
    expanded_df = create_duplicate_rows(df)

    # Step 6: Highlight other mentions in duplicate rows
    print("\nHighlighting other mentions...")
    expanded_df = highlight_other_mentions(expanded_df)

    # Step 7: Save results
    print("\nSaving results...")
    save_dataframe(df, f"{output_prefix}_processed.json", f"{output_prefix}_processed.csv")
    save_dataframe(expanded_df, f"{output_prefix}_expanded.json", f"{output_prefix}_expanded.csv")
    print("Processing complete!\n")


# ------------------------------- #
# Script Execution
# ------------------------------- #

if __name__ == "__main__":
    base_path = "C://Users//kaleb//OneDrive//Desktop//DATA//Paragraphs//Mentions by acronym//"
    
    # Process data for CREC_114
    process_acronym_data(
        files=[f"{base_path}CREC_114_paragraphs_acronym_raw_uuid_UPDATED_ORGS_4-29-2023.json"],
        output_prefix=f"{base_path}CREC_114"
    )

    # Process data for CREC_115
    process_acronym_data(
        files=[f"{base_path}CREC_115_paragraphs_acronym_raw_uuid_UPDATED_ORGS_4-29-2023.json"],
        output_prefix=f"{base_path}CREC_115"
    )
