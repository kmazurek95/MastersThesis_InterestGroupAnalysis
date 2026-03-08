import pandas as pd
import numpy as np
import requests
import logging
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "output"))
INPUT_FILE_PATH = os.path.join(DATA_PATH, "g.references_CREC_114_AND_115.csv")
OUTPUT_FILE_PATH = os.path.join(OUTPUT_PATH, "df_references1.csv")
LOG_FILE = "failed_requests.log"

API_KEY = os.getenv("API_KEY", "YOUR_API_KEY_HERE")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

def construct_package_id(row: pd.Series) -> tuple:
    """
    Create a package ID and corresponding API link for a bill.
    
    Parameters:
    - row: A single row from the DataFrame.

    Returns:
    - A tuple containing:
      1. The package ID for the bill.
      2. The API link to fetch the bill's data.
    """
    bill_version = "is" if row["type"] in ["S", "SRES"] else "ih"

    package_id = f"BILLS-{row['congress']}{row['type'].lower()}{row['number']}{bill_version}"

    link = f"https://api.govinfo.gov/packages/{package_id}/summary?api_key={API_KEY}"

    return package_id, link

def add_package_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'packageId' and 'link' columns to the DataFrame for each bill.
    """
    print("Adding package IDs and API links to the data...")
    df[["packageId", "link"]] = df.apply(construct_package_id, axis=1, result_type="expand")
    return df

def remove_api_key(url: str) -> str:
    """
    Remove the API key from a URL.

    Useful for generating "clean" URLs without sensitive information.
    """
    return url.split("?")[0]

def fetch_bill_data(links: list) -> list:
    """
    Fetch detailed bill data and, if available, the full bill text.
    
    Parameters:
    - links: A list of unique API links for the bills.

    Returns:
    - A list of JSON responses, each containing detailed bill data.
    """
    responses = []

    print("Fetching data for bills...")
    for link in links:
        try:
            url = f"{link}?api_key={API_KEY}"
            print(f"Requesting: {url}")

            response = requests.get(url)
            if response.status_code == 200:
                json_response = response.json()

                txt_link = json_response.get("download", {}).get("txtLink")
                if txt_link:
                    txt_url = f"{txt_link}?api_key={API_KEY}"
                    txt_response = requests.get(txt_url)

                    if txt_response.status_code == 200:
                        json_response["billText"] = txt_response.text
                    else:
                        logging.error(f"Failed to fetch bill text for {txt_url}")
                responses.append(json_response)
            else:
                logging.error(f"Failed request for {url} with status code {response.status_code}")

        except Exception as e:
            logging.error(f"Error processing link {link}: {str(e)}")
    
    return responses

def save_data(df: pd.DataFrame, file_path: str):
    """
    Save a DataFrame to a CSV file.
    """
    print(f"Saving data to: {file_path}")
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    print("\n--- Starting Process ---\n")

    df_references = load_data(INPUT_FILE_PATH)

    df_references = add_package_data(df_references)

    save_data(df_references, OUTPUT_FILE_PATH)

    unique_links = df_references["link"].unique()
    unique_links_without_api_key = [remove_api_key(link) for link in unique_links]

    responses = fetch_bill_data(unique_links_without_api_key)

    print(f"\n--- Process Completed ---\n")
    print(f"Successfully fetched data for {len(responses)} bills.")
    print(f"Failed requests are logged in: {LOG_FILE}")
