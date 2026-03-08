import os
import pandas as pd
import json

DATA_PATH = os.environ.get("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "output"))

with open(os.path.join(DATA_PATH, "CREC 115", "CREC_115_Proccessed_API_Calls", "proccessed_115_bridged.json"), 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
df

num_unique_granule_ids = df['granuleId'].nunique()
print("Number of unique GranuleIds:", num_unique_granule_ids)



columns = df.columns.tolist()
print(', '.join(columns))

print(type(df['members']))
count = df['members'].notna().sum()

print("Number of non-null values in 'members' column:", count)


members_df = pd.DataFrame(df['members'])

members_df = pd.json_normalize(df_members.explode('members')['members'])

print(members_df)
members_df


members_df['unique_id'] = members_df.groupby(['bioGuideId', 'role']).cumcount()

members_wide_df = members_df.pivot_table(index=['bioGuideId', 'unique_id'], columns='role', aggfunc='first').reset_index()

members_wide_df.columns = [f'{y}_{x}' if y != '' else x for x, y in members_wide_df.columns]

members_wide_df
members_wide_df.to_csv(os.path.join(OUTPUT_PATH, "members_wide_df.csv"), index=False, escapechar='\\', sep=',', quotechar='"')


import pandas as pd
from tqdm import tqdm

for item in data:
    if item['committees'] is None:
        item['committees'] = []
    if item['members'] is None:
        item['members'] = []

df = pd.DataFrame(data)

df_expanded = pd.DataFrame()

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    granular_id = index

    members_row = pd.DataFrame(row['members'])
    committees_row = pd.DataFrame(row['committees'])

    members_row['granular_id'] = granular_id
    committees_row['granular_id'] = granular_id

    members_pivoted = members_row.pivot_table(index='granular_id', columns=members_row.groupby('granular_id').cumcount().add(1), aggfunc='first')
    committees_pivoted = committees_row.pivot_table(index='granular_id', columns=committees_row.groupby('granular_id').cumcount().add(1), aggfunc='first')

    members_pivoted.columns = [f'member_{i}_{col}' for i, col in members_pivoted.columns]
    committees_pivoted.columns = [f'committee_{i}_{col}' for i, col in committees_pivoted.columns]

    row_expanded = pd.concat([row.to_frame().transpose(), members_pivoted, committees_pivoted], axis=1)

    df_expanded = pd.concat([df_expanded, row_expanded], ignore_index=True)

df_expanded.set_index(df.index, inplace=True)

df = df_expanded


import requests
import time

api_key = 'YOUR_API_KEY_HERE'  # Get your API key from https://api.data.gov/signup/

url = f"https://api.govinfo.gov/collections/CREC/2017-01-01T00:00:10Z?pageSize=1000&congress=115&offsetMark=%2A&api_key={api_key}"

granule_count_house = 0
granule_count_senate = 0

while url:
    time.sleep(1)
    response = requests.get(url)
    data = response.json()
    print(f"URL: {url}")
    print(f"Status code: {response.status_code}")

    if 'packages' in data:
        for package in data['packages']:
            package_link = package['packageLink']
            package_response = requests.get(f"{package_link}?api_key={api_key}")
            package_data = package_response.json()

            if 'granulesLink' in package_data:
                granules_link = package_data['granulesLink']
                print(f"Package granules link: {granules_link}")

                while granules_link:
                    granules_response = requests.get(f"{granules_link}&api_key={api_key}")
                    granules_data = granules_response.json()

                    for granule in granules_data['granules']:
                        granule_class = granule.get('granuleClass', '')
                        if granule_class == 'HOUSE':
                            granule_count_house += 1
                        elif granule_class == 'SENATE':
                            granule_count_senate += 1

                    granules_link = granules_data.get('nextPage', None)
                    if granules_link:
                        print(f"Next granules page: {granules_link}")

            else:
                print("Granules link not found in package data")

    url = data.get('nextPage', '')

print("House granules:", granule_count_house)
print("Senate granules:", granule_count_senate)



import requests
import time

api_key = 'YOUR_API_KEY_HERE'  # Get your API key from https://api.data.gov/signup/

url = f"https://api.govinfo.gov/collections/CREC/2017-01-01T00:00:10Z?pageSize=1000&congress=115&offsetMark=%2A&api_key={api_key}"

granule_counts = {}

while url:
    time.sleep(1)
    response = requests.get(url)
    data = response.json()
    print(f"URL: {url}")
    print(f"Status code: {response.status_code}")

    if 'packages' in data:
        for package in data['packages']:
            package_id = package['packageId']
            package_link = package['packageLink']
            package_response = requests.get(f"{package_link}?api_key={api_key}")
            package_data = package_response.json()

            granule_count_house = 0
            granule_count_senate = 0

            if 'granulesLink' in package_data:
                granules_link = package_data['granulesLink']
                print(f"Package granules link: {granules_link}")

                while granules_link:
                    granules_response = requests.get(f"{granules_link}&api_key={api_key}")
                    granules_data = granules_response.json()

                    for granule in granules_data['granules']:
                        granule_class = granule.get('granuleClass', '')
                        if granule_class == 'HOUSE':
                            granule_count_house += 1
                        elif granule_class == 'SENATE':
                            granule_count_senate += 1

                    granules_link = granules_data.get('nextPage', None)
                    if granules_link:
                        print(f"Next granules page: {granules_link}")

                granule_counts[package_id] = {
                    'House': granule_count_house,
                    'Senate': granule_count_senate
                }
            else:
                print("Granules link not found in package data")

    url = data.get('nextPage', '')

print("Granule counts for each package:")
for package_id, counts in granule_counts.items():
    print(f"Package {package_id}: House granules: {counts['House']}, Senate granules: {counts['Senate']}")


total_house_granules = 0
total_senate_granules = 0

for package_id, counts in granule_counts.items():
    total_house_granules += counts['House']
    total_senate_granules += counts['Senate']

print(f"Total House granules: {total_house_granules}")
print(f"Total Senate granules: {total_senate_granules}")


import pandas as pd
import json
from tqdm import tqdm

with open(os.path.join(DATA_PATH, "Raw API Calls", "final_result_115a.json"), 'r') as f:
    data = [json.loads(line) for line in f]


import pandas as pd
import json

with open(os.path.join(DATA_PATH, "Raw API Calls", "final_result_115a.json"), 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)


unique_granule_ids = df['granuleId'].nunique()
unique_package_ids = df['packageId'].nunique()
print(f"Number of unique granule ids: {unique_granule_ids}")
print(f"Number of unique package ids: {unique_package_ids}")

