import os
import json
import pandas as pd

# ========================= Utility Functions =========================

def load_json_lines(file_path):
    """
    Load a JSON lines file into a pandas DataFrame.

    Args:
        file_path (str): Path to the JSON lines file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def save_dataframe(df, output_path, file_name):
    """
    Save a pandas DataFrame to CSV and JSON formats.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Directory to save the files.
        file_name (str): Base name of the files (without extension).
    """
    os.makedirs(output_path, exist_ok=True)

    # Save as CSV
    csv_path = os.path.join(output_path, f"{file_name}.csv")
    df.to_csv(csv_path, encoding='utf-8', index=False)

    # Save as JSON lines
    json_path = os.path.join(output_path, f"{file_name}.json")
    df.to_json(json_path, orient='records', lines=True, force_ascii=False)

    print(f"✅ Saved CSV: {csv_path}")
    print(f"✅ Saved JSON: {json_path}")


def filter_json_data(json_file_path, mention_index, acronym, variations_to_exclude):
    """
    Filter JSON data based on specified criteria.

    Args:
        json_file_path (str): Path to the JSON lines file.
        mention_index (int or None): Mention index to filter (or None to skip filtering by mention index).
        acronym (int): Acronym filter value (0 for names, 1 for acronyms).
        variations_to_exclude (list): List of variations to exclude from the data.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Load data
    df = load_json_lines(json_file_path)
    original_size = len(df)

    # Apply filters
    if mention_index is None:
        df_filtered = df[df['acronym'] == acronym]
    else:
        df_filtered = df[(df['mention_index'] == mention_index) & (df['acronym'] == acronym)]
    
    # Exclude specific variations
    df_filtered = df_filtered[~df_filtered['variation'].isin(variations_to_exclude)]
    
    # Log filtering results
    filtered_size = len(df_filtered)
    print(f"Original rows: {original_size}, Filtered rows: {filtered_size}, Dropped rows: {original_size - filtered_size}")

    return df_filtered


def merge_predictions(dataframe, predictions_path):
    """
    Merge predictions into the mentions DataFrame based on `uuid_paragraph`.

    Args:
        dataframe (pd.DataFrame): Mentions DataFrame to merge with predictions.
        predictions_path (str): Path to the predictions JSON file.

    Returns:
        pd.DataFrame: Merged DataFrame containing predictions.
    """
    # Load predictions JSON into a DataFrame
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['prediction']).reset_index()
    df_predictions.rename(columns={'index': 'uuid_paragraph'}, inplace=True)
    
    # Merge predictions with the mentions DataFrame
    merged_df = pd.merge(dataframe, df_predictions, on='uuid_paragraph', how='inner')
    merged_df.rename(columns={'prediction': 'prominence'}, inplace=True)

    return merged_df

# ========================= Main Script =========================

def main():
    """
    Main script to filter paragraphs, merge predictions, and save the results.
    """
    # File paths and configurations
    json_file_path = r'C://Users//kaleb//OneDrive//Desktop//DATA//Paragraphs//3B_____PARAGRAPHS_name_AND_acronym_114_AND_115_EXPANDED__UPDATED__4-29-2023____1.json'
    predictions_path = r'C://Users//kaleb//OneDrive//Desktop//Classifier//Model//predictions.json'
    output_path = r'C://Users//kaleb//OneDrive//Desktop//OUTPUT DATA'
    output_file_name = 'paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B'

    # Filter parameters
    mention_index = None  # Set to filter by mention index, or None to skip
    acronym = 0  # 0 for names, 1 for acronyms
    variations_to_exclude = [
        'National Labor Relations Board', 'Continuum of Care', 'Political action committee',
        'Small Business and Entrepreneurship', 'Mentor', "America's Promise",
        'Reproductive technology', 'Trade association', 'Freedom Project, The',
        'Indian Gaming Regulatory Act'
    ]

    # Step 1: Filter paragraphs data
    print("\n=== Step 1: Filtering Paragraphs Data ===")
    df_mentions = filter_json_data(json_file_path, mention_index, acronym, variations_to_exclude)

    # Step 2: Merge filtered data with predictions
    print("\n=== Step 2: Merging Data with Predictions ===")
    df_classified = merge_predictions(df_mentions, predictions_path)

    # Step 3: Save results
    print("\n=== Step 3: Saving Classified Data ===")
    save_dataframe(df_classified, output_path, output_file_name)

    # Summary
    print(f"\n🎉 Processing Complete! Total rows classified: {len(df_classified)}")

if __name__ == "__main__":
    main()
