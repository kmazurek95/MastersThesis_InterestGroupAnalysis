# Install necessary libraries (use only if not installed)
!pip install requests beautifulsoup4 tqdm retrying pandas

# Imports
import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from retrying import retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your API key (replace with your actual key)
os.environ['GOVINFO_API_KEY'] = 'IgmhiBjvtZhuBDFLlFSYkuGucsP2wKc8it97N4ln'

# -------------------------------
# Helper Functions
# -------------------------------

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def make_request(url, params=None):
    """
    Send an API request with retry logic in case of failures.
    """
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        raise

def read_progress(filename):
    """
    Retrieve saved progress from a file to resume work.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def store_progress(filename, progress):
    """
    Save progress to a file to ensure continuity.
    """
    with open(filename, 'w') as file:
        json.dump(progress, file)

def extract_text_from_html(soup):
    """
    Extract readable text content from HTML.
    """
    return soup.get_text(separator=' ').strip()

def fetch_granule_details(link, api_key):
    """
    Fetch and parse details for a single granule, including text content.
    """
    try:
        response = make_request(f"{link}?api_key={api_key}")
        download_link = response.get("download", {}).get("txtLink")
        if download_link:
            html_response = requests.get(download_link)
            if html_response.status_code == 200:
                soup = BeautifulSoup(html_response.text, "html.parser")
                response["parsed_text"] = extract_text_from_html(soup)
        return response
    except Exception as e:
        print(f"Error fetching granule details for {link}: {e}")
        return None

# -------------------------------
# API Data Fetching Logic
# -------------------------------

def fetch_packages(api_url, params):
    """
    Retrieve package data from the API.
    """
    packages = []
    while api_url:
        print(f"Fetching package data from {api_url}...")
        response = make_request(api_url, params)
        packages.extend(response.get("packages", []))
        api_url = response.get("nextPage")
        params = None  # Params are only required for the first request
    print(f"Total packages retrieved: {len(packages)}")
    return packages

def fetch_granules(granules_url, api_key):
    """
    Retrieve links to granules for a given package.
    """
    granule_links = []
    while granules_url:
        try:
            response = make_request(f"{granules_url}&api_key={api_key}")
            granule_links.extend(granule["granuleLink"] for granule in response.get("granules", []))
            granules_url = response.get("nextPage")
        except Exception as e:
            print(f"Error fetching granules: {e}")
            break
    return granule_links

def process_granules_parallel(granule_links, api_key, workers=8):
    """
    Fetch granule details concurrently for better performance.
    """
    results = []
    print("Processing granules in parallel...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_granule_details, link, api_key): link for link in granule_links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Granules Progress"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing granule: {e}")
    return results

# -------------------------------
# Main Functionality
# -------------------------------

def main():
    """
    Main function to fetch data for the 114th and 115th Congresses.
    """
    congresses = [114, 115]  # List of congresses to fetch data for
    progress_file = "progress.json"  # File to track progress
    progress = read_progress(progress_file)
    
    all_dataframes = []

    for congress in congresses:
        print(f"\nStarting data collection for the {congress}th Congress...")

        # API endpoint and parameters
        api_url = "https://api.govinfo.gov/collections/CREC/2015-01-01T00:00:10Z"
        params = {
            "pageSize": 1000,
            "congress": congress,
            "offsetMark": "*",
            "api_key": os.getenv("GOVINFO_API_KEY"),
        }

        # Resume from saved progress if available
        start_index = progress.get(str(congress), 0)

        # Fetch packages
        packages = fetch_packages(api_url, params)

        for idx, package in enumerate(packages[start_index:], start=start_index):
            try:
                print(f"\nProcessing package {idx + 1}/{len(packages)} for the {congress}th Congress...")
                package_url = f"{package['packageLink']}?api_key={params['api_key']}"
                package_data = make_request(package_url)

                granules_url = package_data.get("granulesLink")
                if not granules_url:
                    print(f"No granules found for package {idx + 1}. Skipping...")
                    continue

                granule_links = fetch_granules(granules_url, params["api_key"])
                print(f"Found {len(granule_links)} granules.")

                # Process granules concurrently
                granule_data = process_granules_parallel(granule_links, params["api_key"])
                df = pd.DataFrame(granule_data)
                all_dataframes.append(df)

                # Save intermediate progress
                progress[str(congress)] = idx + 1
                store_progress(progress_file, progress)
                df.to_csv(f"package_{congress}_{idx}.csv", index=False)

            except Exception as e:
                print(f"Error processing package {idx + 1} for the {congress}th Congress: {e}")
                continue

    # Combine and save final results
    if all_dataframes:
        final_result = pd.concat(all_dataframes, ignore_index=True)
        final_result.to_csv("final_results_congresses.csv", index=False)
        final_result.to_json("final_results_congresses.json", orient="records", lines=True)
        print("\nData collection completed. Final results saved.")

if __name__ == "__main__":
    main()
