import os
import json
import pandas as pd
from typing import List


# ========================= Configuration =========================

# Input and output paths
DATA_PATH = "C://Users//kaleb//OneDrive//Desktop//DATA//Paragraphs//"
OUTPUT_PATH = "C:/Users/kaleb/OneDrive/Desktop/OUTPUT DATA/"
LABELED_DATA_PATH = "C:/Users/kaleb/OneDrive/Desktop/Classifier/"

# Exclusion lists
VARIATIONS_TO_EXCLUDE_NAME = [
    'National Labor Relations Board', 'Continuum of Care', 'Political action committee',
    'Small Business and Entrepreneurship', 'Mentor', "America's Promise", 'Reproductive technology',
    'Trade association', 'Freedom Project, The', 'Indian Gaming Regulatory Act'
]

VARIATIONS_TO_EXCLUDE_ACRONYM = [
    'ACT', 'Brady', 'ACA', 'AIR', 'CARE', 'NDAA', 'CRA', 'NRA', 'CARA', 'CAA', 'AAA', 'ADA', 'COST',
    'FERC', 'AF', 'OPP', 'SAFE', 'IDEA', 'NSA', 'AA', 'ISSA', 'NETWORK', 'ABA', 'PAC', 'NLRB', 'AMS',
    'MAP', 'SSA', 'GSA', 'NCAA', 'AIM', 'FAIR', 'APS', 'ABC', 'NAS', 'NSF', 'ATF', 'AMT', 'SEA', 'IFA',
    'AFB', 'CPI', 'OCC', 'ESA', 'ARC', 'RPA', 'CASE', 'NBA', 'PASS', 'ASIA', 'NOW', 'CAP', 'PRC', 'AAR',
    'FRA', 'NPR', 'IHS', 'NSC', 'APA', 'PER', 'LISA', 'ACE', 'NGA', 'CWA', 'DAM', 'CCA', 'SMART', 'MDA',
    'ATS', 'FTA', 'MSC', 'GAP', 'UNESCO', 'CBA', 'RISE', 'ADS', 'MICA', 'AAG', 'EA', 'AGI', 'SAA', 'NTF',
    'CORE'
]


# ========================= Utility Functions =========================

def read_json_as_dataframe(json_path: str) -> pd.DataFrame:
    """
    Reads a JSON file line by line into a Pandas DataFrame.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    data = []
    with open(json_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def filter_dataframe(df: pd.DataFrame, mention_index: int, acronym: int, variations_to_exclude: List[str]) -> pd.DataFrame:
    """
    Filters a DataFrame based on mention index, acronym, and exclusion list.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        mention_index (int): Mention index to filter.
        acronym (int): Acronym flag (0 or 1).
        variations_to_exclude (List[str]): List of variations to exclude.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    original_size = len(df)
    df_filtered = df[(df['mention_index'] == mention_index) & (df['acronym'] == acronym)]
    df_filtered = df_filtered[~df_filtered['variation'].isin(variations_to_exclude)]

    dropped_rows = original_size - len(df_filtered)
    print(f"Original rows: {original_size}, Filtered rows: {len(df_filtered)}, Dropped rows: {dropped_rows}")
    return df_filtered


def check_overlap(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> set:
    """
    Identifies overlapping values in a specific column between two DataFrames.
    
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        column (str): Column to check for overlap.
        
    Returns:
        set: Set of overlapping values.
    """
    overlap = set(df1[column]).intersection(set(df2[column]))
    print(f"Number of overlapping values: {len(overlap)}")
    return overlap


def save_data(df: pd.DataFrame, output_dir: str, file_name: str):
    """
    Saves a DataFrame to both JSON and CSV formats.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        output_dir (str): Directory to save files.
        file_name (str): Base name of the file.
    """
    output_json = os.path.join(output_dir, f"{file_name}.json")
    output_csv = os.path.join(output_dir, f"{file_name}.csv")

    # Save as JSON
    with open(output_json, 'w', encoding='utf-8') as file:
        for record in df.to_dict(orient='records'):
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ Saved JSON: {output_json}")

    # Save as CSV
    df.to_csv(output_csv, index=False, escapechar='\\', sep=',', quotechar='"')
    print(f"✅ Saved CSV: {output_csv}")


# ========================= Main Processing Workflow =========================

def main():
    """Main script to filter and prepare data for labeling."""
    
    # Step 1: Load the main JSON data
    print("\n=== Step 1: Loading Data ===")
    json_file = os.path.join(DATA_PATH, "3B_____PARAGRAPHS_name_AND_acronym_114_AND_115_EXPANDED__UPDATED__4-29-2023____1.json")
    df = read_json_as_dataframe(json_file)
    print(f"Data loaded. Total rows: {len(df)}")

    # Step 2: Filter data for names and acronyms
    print("\n=== Step 2: Filtering Data ===")
    print("Filtering NAME data...")
    df_name_filtered = filter_dataframe(df, mention_index=1, acronym=0, variations_to_exclude=VARIATIONS_TO_EXCLUDE_NAME)

    print("Filtering ACRONYM data...")
    df_acronym_filtered = filter_dataframe(df, mention_index=1, acronym=1, variations_to_exclude=VARIATIONS_TO_EXCLUDE_ACRONYM)

    # Step 3: Random sampling for labeling
    print("\n=== Step 3: Random Sampling ===")
    df_name_sample = df_name_filtered.sample(n=350, random_state=98)
    df_acronym_sample = df_acronym_filtered.sample(n=350, random_state=98)

    # Step 4: Remove overlap with existing labeled data
    print("\n=== Step 4: Removing Overlap ===")
    labeled_data = pd.read_csv(os.path.join(LABELED_DATA_PATH, "LABELING__UPDATED__4-26-2023____Sample__2.csv"))
    df_name_sample = df_name_sample[~df_name_sample['uuid_paragraph'].isin(check_overlap(df_name_sample, labeled_data, 'uuid_paragraph'))]
    df_acronym_sample = df_acronym_sample[~df_acronym_sample['uuid_paragraph'].isin(check_overlap(df_acronym_sample, labeled_data, 'uuid_paragraph'))]

    # Step 5: Save the filtered samples
    print("\n=== Step 5: Saving Filtered Samples ===")
    save_data(df_name_sample, LABELED_DATA_PATH, "LABELING__UPDATED__4-29-2023____Sample__3A")
    save_data(df_acronym_sample, LABELED_DATA_PATH, "LABELING__UPDATED__4-29-2023____Sample__3B")

    # Step 6: Export unlabeled data for names
    print("\n=== Step 6: Exporting Unlabeled NAME Data ===")
    columns_to_keep = ['p1_original', 'uuid_paragraph']
    df_name_unlabeled = df_name_filtered[columns_to_keep]
    df_name_unlabeled.to_csv(os.path.join(OUTPUT_PATH, "UNLABELED_DATA___Paragraphs_NAME_114_115.csv"), index=False)
    print(f"✅ Unlabeled NAME data exported. Rows: {df_name_unlabeled.shape[0]}")

    print("\n🎉 Processing Complete!")


# ========================= Script Execution =========================

if __name__ == "__main__":
    main()
