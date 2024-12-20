import os
import json
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Utility Functions
# ---------------------------

def remove_duplicates(df, subset_cols):
    """
    Remove duplicate rows based on specified columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        subset_cols (list): List of column names to check for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    duplicates = df.duplicated(subset=subset_cols)
    print(f"Number of duplicate rows removed: {duplicates.sum()}")
    return df.drop_duplicates(subset=subset_cols)


def aggregate_uuid_paragraph(df, group_col, uuid_col):
    """
    Aggregate UUIDs by paragraph and calculate overlap (if multiple UUIDs exist).
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): Column to group by (e.g., 'paragraph').
        uuid_col (str): Column containing UUIDs.

    Returns:
        pd.DataFrame: Aggregated DataFrame with overlap information.
    """
    def aggregator(group):
        unique_ids = group[uuid_col].unique()
        overlap_ids = unique_ids.tolist() if len(unique_ids) > 1 else []
        return pd.Series({
            'overlap_ids': overlap_ids,
            'overlap_count': len(overlap_ids)
        })

    return df.groupby(group_col).apply(aggregator).reset_index()


def highlight_mentions(text, word):
    """
    Highlight all mentions of a word in a given text.
    
    Args:
        text (str): The input text.
        word (str): The word to highlight.
    
    Returns:
        str: Text with highlighted mentions of the word.
    """
    mentions = list(re.finditer(re.escape(word), text, flags=re.IGNORECASE))
    if not mentions:
        return text

    highlighted_text = []
    current_pos = 0

    for idx, match in enumerate(mentions, start=1):
        start, end = match.start(), match.end()
        highlighted_word = f"****({idx}){text[start:end]}****"
        highlighted_text.append(text[current_pos:start] + highlighted_word)
        current_pos = end

    highlighted_text.append(text[current_pos:])
    return ''.join(highlighted_text)


def clean_paragraph(paragraph):
    """
    Remove metadata from paragraphs, specifically text before 'www.gpo.gov'.
    
    Args:
        paragraph (str): The input paragraph.
    
    Returns:
        str: Cleaned paragraph.
    """
    parts = paragraph.split('www.gpo.gov')
    if len(parts) > 1:
        closing_bracket = parts[-1].find(']')
        return parts[-1][closing_bracket + 1:].strip() if closing_bracket != -1 else parts[-1].strip()
    return paragraph


def find_similar_paragraphs(df, text_col, group_cols, threshold=0.5):
    """
    Identify similar paragraphs within grouped data using TF-IDF and cosine similarity.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_col (str): Column containing the text to compare.
        group_cols (list): Columns to group data by.
        threshold (float): Cosine similarity threshold for identifying similar paragraphs.
    
    Returns:
        pd.DataFrame: DataFrame of similar paragraph pairs with similarity scores.
    """
    similar_pairs = []
    groups = df.groupby(group_cols)

    for group_key, group_df in tqdm(groups, desc="Processing Groups"):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(group_df[text_col])
        cosine_sim = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(cosine_sim, 0)  # Ignore self-similarity

        # Identify paragraph pairs exceeding the similarity threshold
        pairs = np.argwhere(cosine_sim > threshold)
        for (i, j) in pairs:
            if i < j:  # Avoid duplicate pairings
                similar_pairs.append({
                    'paragraph_id1': group_df.iloc[i]['uuid_paragraph'],
                    'paragraph_id2': group_df.iloc[j]['uuid_paragraph'],
                    'paragraph1': group_df.iloc[i][text_col],
                    'paragraph2': group_df.iloc[j][text_col],
                    'similarity_score': cosine_sim[i, j],
                    'granuleId': group_df.iloc[i]['granuleId'],
                    'txt_link1': group_df.iloc[i]['txt_link'],
                    'txt_link2': group_df.iloc[j]['txt_link']
                })

    return pd.DataFrame(similar_pairs)

# ---------------------------
# Main Processing
# ---------------------------

if __name__ == "__main__":
    # Load Data
    print("Loading data...")
    input_file = 'C://Users//kaleb//OneDrive//Desktop//Mentions//CREC_115//Mentions_Name_115//Data_Input_and_Final_Output//CREC_115_paragraphs_raw_uuid.json'
    df = pd.read_json(input_file, lines=True)

    # Step 1: Remove duplicates
    print("Removing duplicate rows...")
    df = remove_duplicates(df, subset_cols=['org_id', 'paragraph'])

    # Step 2: Aggregate UUIDs by paragraph
    print("Aggregating UUIDs by paragraph...")
    aggregated_data = aggregate_uuid_paragraph(df, group_col='paragraph', uuid_col='uuid_paragraph')
    df = df.merge(aggregated_data, on='paragraph', how='left')

    # Step 3: Clean paragraphs
    print("Cleaning paragraphs...")
    tqdm.pandas(desc="Cleaning paragraphs")
    df['cleaned_paragraph'] = df['paragraph'].progress_apply(clean_paragraph)

    # Step 4: Highlight mentions in paragraphs
    print("Highlighting mentions in paragraphs...")
    tqdm.pandas(desc="Highlighting mentions")
    df['highlighted_paragraph'] = df.apply(
        lambda row: highlight_mentions(row['cleaned_paragraph'], row['variation']),
        axis=1
    )

    # Step 5: Find similar paragraphs
    print("Identifying similar paragraphs...")
    similarity_threshold = 0.5
    similar_paragraphs = find_similar_paragraphs(
        df,
        text_col='highlighted_paragraph',
        group_cols=['granuleId'],
        threshold=similarity_threshold
    )

    # Save Results
    print("Saving results...")
    output_file = 'C://Users//kaleb//OneDrive//Desktop//Processed_Data//similar_paragraphs.csv'
    similar_paragraphs.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")
