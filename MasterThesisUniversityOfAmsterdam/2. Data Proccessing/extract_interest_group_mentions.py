import os
import re
import json
import pandas as pd
import nltk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

# Download NLTK's sentence tokenizer
nltk.download('punkt')

# Configure logging to track progress and errors
logging.basicConfig(
    filename="progress.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -------------------------
# Utility Functions
# -------------------------

def read_json_file(file_path):
    """
    Load a JSON file containing multiple records into a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_to_json(df, file_path, orient='records', lines=True):
    """
    Save a DataFrame as a JSON file.
    """
    df.to_json(file_path, orient=orient, lines=lines, force_ascii=False)


def save_to_csv(df, file_path):
    """
    Save a DataFrame as a CSV file.
    """
    df.to_csv(file_path, index=False, escapechar='\\', sep=',', quotechar='"')


def extract_interest_group_paragraphs(text, interest_groups, granule_id, txt_link):
    """
    Identify and extract paragraphs mentioning interest groups from the text.
    """
    extracted = []
    acronyms = []
    if not text:
        return extracted, acronyms

    sentences = nltk.sent_tokenize(text)
    unique_mentions = set()

    for org_id, interest_group, original_name, current_name, acronym, name_match in interest_groups:
        # Generate variations for interest group mentions (e.g., names and acronyms)
        variations = [(original_name, False), (current_name, False), (acronym, True)] if current_name else [(original_name, False), (acronym, True)]
        
        # Search for each variation in the sentences
        for i, sentence in enumerate(sentences):
            for variation, is_acronym in variations:
                if not isinstance(variation, str):  # Skip invalid variations
                    continue

                # Match the variation in the text using regular expressions
                pattern = re.compile(rf'\b{re.escape(variation)}\b', re.IGNORECASE if not is_acronym else 0)
                if pattern.search(sentence):
                    # Combine surrounding sentences to create a paragraph
                    start, end = max(0, i - 3), min(len(sentences), i + 4)
                    if any(idx in unique_mentions for idx in range(start, end)):
                        continue  # Skip already-processed mentions

                    paragraph = " ".join(sentences[start:end])
                    unique_mentions.update(range(start, end))
                    
                    # Prepare the extracted data
                    data = {
                        'org_id': org_id,
                        'interest_group': interest_group,
                        'original_name': original_name,
                        'current_name': current_name,
                        'variation': variation,
                        'granuleId': granule_id,
                        'paragraph': paragraph,
                        'txt_link': txt_link
                    }
                    (acronyms if is_acronym else extracted).append(data)

    return extracted, acronyms


def process_row(row, interest_groups):
    """
    Process a single row of data to extract interest group paragraphs.
    """
    # Extract the text link from the row
    txt_link = re.search(r'href="([^"]*)"', row.get('txt_link', '')).group(1) if 'txt_link' in row and isinstance(row['txt_link'], str) else ""
    return extract_interest_group_paragraphs(row.get('parsed_content_text', ''), interest_groups, row['granuleId'], txt_link)


# -------------------------
# Core Processing Function
# -------------------------

def process_dataset(dataset_file, processed_file, interest_groups_file, output_prefix, num_threads=6, chunk_size=100):
    """
    Process a dataset to extract paragraphs mentioning interest groups.
    """
    # Step 1: Load and filter dataset
    df1 = pd.DataFrame(read_json_file(dataset_file))
    df1_filtered = df1[(df1['mention_index'] == 1) & (df1['paragraph_mention_count'] > 1)]

    # Extract unique organization and granule IDs
    unique_org_ids = df1_filtered['org_id'].unique()
    unique_granule_ids = df1_filtered['granuleId'].unique()

    # Step 2: Load processed data and filter relevant granule IDs
    filtered_chunks = []
    for chunk in pd.read_json(processed_file, lines=True, chunksize=5000):
        filtered_chunks.append(chunk[chunk['granuleId'].isin(unique_granule_ids)])
    df2_filtered = pd.concat(filtered_chunks)

    # Step 3: Load interest group information
    df_interest_groups = pd.read_csv(interest_groups_file)
    df_interest_groups['current_name_2'].fillna(df_interest_groups['original_name_2'], inplace=True)
    df_interest_groups[['original_name_2', 'current_name_2', 'acronym_2']] = df_interest_groups[['original_name_2', 'current_name_2', 'acronym_2']].apply(lambda col: col.str.strip())
    interest_groups = df_interest_groups.to_records(index=False)

    # Step 4: Initialize result tracking
    result_data, acronym_data = [], []
    result_counter, acronym_counter = 0, 0

    # Step 5: Process dataset in chunks using multithreading
    total_chunks = len(df2_filtered) // chunk_size + (1 if len(df2_filtered) % chunk_size > 0 else 0)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for idx in tqdm(range(total_chunks), desc=f"Processing {output_prefix}"):
            chunk = df2_filtered.iloc[idx * chunk_size: (idx + 1) * chunk_size]
            future = executor.submit(lambda c: [process_row(row, interest_groups) for _, row in c.iterrows()], chunk)
            results = future.result()

            for res, acronyms in results:
                result_data.extend(res)
                acronym_data.extend(acronyms)

            # Save progress every 10 chunks
            if (idx + 1) % 10 == 0:
                save_to_json(pd.DataFrame(result_data), f"{output_prefix}_temp_results_{result_counter:03d}.json")
                save_to_json(pd.DataFrame(acronym_data), f"{output_prefix}_temp_acronyms_{acronym_counter:03d}.json")
                result_counter += 1
                acronym_counter += 1
                result_data, acronym_data = [], []

    # Final save
    save_to_json(pd.DataFrame(result_data), f"{output_prefix}_results.json")
    save_to_json(pd.DataFrame(acronym_data), f"{output_prefix}_acronyms.json")


# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    # Process datasets for CREC 114 and 115
    process_dataset(
        dataset_file="/path/to/CREC_114_PARAGRAPHS.json",
        processed_file="/path/to/CREC_114_PROCESSED.json",
        interest_groups_file="/path/to/interest_group_info.csv",
        output_prefix="CREC_114",
    )
    process_dataset(
        dataset_file="/path/to/CREC_115_PARAGRAPHS.json",
        processed_file="/path/to/CREC_115_PROCESSED.json",
        interest_groups_file="/path/to/interest_group_info.csv",
        output_prefix="CREC_115",
    )
