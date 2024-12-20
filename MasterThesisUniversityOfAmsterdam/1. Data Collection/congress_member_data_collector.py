import pandas as pd
import json
import requests
from tqdm import tqdm

# ========================= Configuration =========================

# File Paths and API Configuration
CSV_PATH = "g.members_CREC_114_AND_115.csv"       # Input CSV containing bioguide IDs
API_KEY = "IgmhiBjvtZhuBDFLlFSYkuGucsP2wKc8it97N4ln"  # Replace with your Congress API key
LOG_FILE = "failed_requests.log"                 # Log file for failed requests
OUTPUT_JSON = "responses_pretty.json"            # File to save raw API responses
OUTPUT_CSV = "bio_data.csv"                      # File to save cleaned data

# API Base URL
BASE_URL = "https://api.congress.gov/v3/member"

# ========================= Utility Functions =========================

def load_csv(file_path):
    """
    Load a CSV file into a DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def save_json(data, file_path, pretty=False):
    """
    Save data as a JSON file.

    Args:
        data (list or dict): Data to save.
        file_path (str): Path to the output JSON file.
        pretty (bool): Whether to pretty-print the JSON.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=4)
        else:
            json.dump(data, f)
    print(f"Saved JSON to {file_path}")

def fetch_member_data(bioguide_ids, api_key, log_file):
    """
    Fetch data from the Congress API for a list of bioguide IDs.

    Args:
        bioguide_ids (list): List of bioguide IDs.
        api_key (str): API key for authentication.
        log_file (str): Path to log failed requests.
    
    Returns:
        list: A list of successful API responses.
    """
    responses = []
    with open(log_file, "w") as log:
        for bioguide_id in tqdm(bioguide_ids, desc="Fetching member data"):
            url = f"{BASE_URL}/{bioguide_id}?format=json&api_key={api_key}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    responses.append(response.json())
                else:
                    log.write(f"{bioguide_id}: Error {response.status_code}\n")
            except Exception as e:
                log.write(f"{bioguide_id}: Exception {str(e)}\n")
    print(f"API requests completed. Failed requests logged to {log_file}.")
    return responses

def process_member_data(data):
    """
    Process nested JSON member data into a flattened DataFrame.

    Args:
        data (list): List of raw API responses.
    
    Returns:
        pd.DataFrame: Cleaned and flattened DataFrame.
    """
    if not data:
        print("No data to process.")
        return pd.DataFrame()

    # Normalize JSON into a flat DataFrame
    df = pd.json_normalize(data)

    # Extract and clean required fields
    df['birthYear'] = df['member.birthYear']
    df['cosponsoredLegislationCount'] = df['member.cosponsoredLegislation.count']
    df['partyHistory'] = df['member.partyHistory'].apply(
        lambda x: x[0]['partyName'] if isinstance(x, list) and x else None
    )

    # Handle nested 'terms' field
    print("Exploding and normalizing terms data...")
    df = df.explode('member.terms').reset_index(drop=True)
    terms_df = pd.json_normalize(df['member.terms'])

    # Merge terms back into the main DataFrame
    df = pd.concat([df, terms_df], axis=1).drop(columns=['member.terms'])

    # Filter rows for relevant congress sessions (114 and 115)
    df = df[df['congress'].isin([114, 115])]

    # Rename columns for clarity
    df = df.rename(columns={'member.identifiers.bioguideId': 'bioGuideId'})
    df['i.bioGuideId'] = df['bioGuideId']  # Duplicate column as required

    print("Data processing complete.")
    return df

# ========================= Main Script =========================

def main():
    """Main script to fetch, process, and save congress member data."""
    print("\n=== Step 1: Loading Bioguide IDs ===")
    df_references = load_csv(CSV_PATH)
    if df_references.empty:
        print("No data found in the input CSV. Exiting.")
        return

    # Extract unique bioguide IDs
    unique_bioguide_ids = df_references['bioGuideId'].dropna().unique().tolist()
    print(f"Total unique bioguide IDs: {len(unique_bioguide_ids)}")

    print("\n=== Step 2: Fetching Member Data from API ===")
    member_data = fetch_member_data(unique_bioguide_ids, API_KEY, LOG_FILE)

    print("\n=== Step 3: Saving Raw JSON Responses ===")
    save_json(member_data, OUTPUT_JSON, pretty=True)

    print("\n=== Step 4: Processing Data into a Cleaned Format ===")
    processed_df = process_member_data(member_data)

    print("\n=== Step 5: Saving Cleaned Data to CSV ===")
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cleaned data saved to {OUTPUT_CSV}")

    print("\n=== Step 6: Summary Statistics ===")
    print(f"Processed rows: {processed_df.shape[0]}")
    print(f"Failed requests log saved to: {LOG_FILE}")
    print(f"Raw responses saved to: {OUTPUT_JSON}")
    print(f"Final cleaned data saved to: {OUTPUT_CSV}")

# ========================= Script Execution =========================

if __name__ == "__main__":
    main()
