import pandas as pd
import os

# =========================== Utility Functions ============================

def load_dataframe(file_path, file_type='csv'):
    """
    Load a CSV or JSON file into a DataFrame.
    
    Parameters:
        file_path (str): Path to the input file.
        file_type (str): Type of file ('csv' or 'json').
    
    Returns:
        pd.DataFrame: Loaded DataFrame or empty DataFrame if the file is not found.
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_type == 'json':
            return pd.read_json(file_path, orient='records', lines=True, encoding='utf-8')
        else:
            raise ValueError("Invalid file type. Use 'csv' or 'json'.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def display_directory_files(directory):
    """
    Print all files in a given directory.
    
    Parameters:
        directory (str): Path to the directory.
    """
    print("Files in directory:")
    for file_name in os.listdir(directory):
        print(f" - {file_name}")

def calculate_mode(series):
    """
    Calculate the mode of a pandas Series, with a tiebreak to return the first mode in case of ties.
    
    Parameters:
        series (pd.Series): Series for which to calculate the mode.
    
    Returns:
        value: First mode, or None if the series is empty.
    """
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else None

# =========================== File Paths and Loading ============================

# Define the base directory and file paths
BASE_DIR = r'C:\Users\kaleb\OneDrive\Desktop\DATA\COMPLETE\\'
committees_file = os.path.join(BASE_DIR, 'g.committees_CREC_114_AND_115.csv')
prominence_file = os.path.join(BASE_DIR, 'df_interest_groups_prominence.csv')

# Display available files in the directory
display_directory_files(BASE_DIR)

# Load data
print("\nLoading data...")
df_committees = load_dataframe(committees_file, file_type='csv')
df_prominence = load_dataframe(prominence_file, file_type='csv')

# =========================== Committees Data Analysis ============================

if not df_committees.empty:
    print("\n### Committees Data Analysis ###")
    
    # Display columns in the committees DataFrame
    print("\nColumns in Committees DataFrame:")
    print(df_committees.columns)

    # 1. Count 'granuleId' with only one unique 'committeeName'
    single_committee_count = (
        df_committees.groupby('granuleId')['committeeName'].nunique().eq(1).sum()
    )
    print(f"\nNumber of 'granuleId' with only one unique 'committeeName': {single_committee_count}")

    # 2. Frequency distribution of 'granuleId' occurrences
    granule_frequency = df_committees['granuleId'].value_counts()
    frequency_distribution = granule_frequency.value_counts().sort_index()
    print("\nFrequency distribution of 'granuleId' occurrences:")
    print(frequency_distribution)

    # 3. Unique committee names
    unique_committees = df_committees['committeeName'].unique()
    print("\nUnique committee names in the data:")
    print(unique_committees)

    # 4. Policy Area Mapping
    committee_to_policy_area =  {
    'Committee on Banking, Housing, and Urban Affairs': ['Housing', 1400],
    'Committee on the Judiciary': ['Law and Crime', 1200],
    'Committee on Health, Education, Labor, and Pensions': ['Health', 300],
    'Committee on Appropriations': ['Macroeconomics', 100],
    'Committee on Agriculture, Nutrition, and Forestry': ['Agriculture', 400],
    'Committee on Foreign Relations': ['International Affairs', 1900],
    'Committee on Rules': ['Government Operations', 2000],
    'Committee on Agriculture': ['Agriculture', 400],
    'Committee on Education and the Workforce': ['Education', 600],
    'Committee on Foreign Affairs': ['International Affairs', 1900],
    'Committee on Education and Labor': ['Education', 600],
    'Committee on Environment and Public Works': ['Environment', 700],
    "Committee on Veterans' Affairs": ['Government Operations', 2000],
    'Committee on Armed Services': ['Defense', 1600],
    'Committee on Energy and Natural Resources': ['Energy', 800],
    'Committee on Finance': ['Macroeconomics', 100],
    'Committee on Energy and Commerce': ['Energy', 800],
    'Committee on Ways and Means': ['Macroeconomics', 100],
    'Committee on Small Business and Entrepreneurship': ['Domestic Commerce', 1500],
    'Committee on Homeland Security and Governmental Affairs': ['Government Operations', 2000],
    'Committee on Financial Services': ['Domestic Commerce', 1500],
    'Joint Committee on Taxation': ['Macroeconomics', 100],
    'Committee on Natural Resources': ['Public Lands', 2100],
    'Permanent Select Committee on Intelligence': ['Defense', 1600],
    'Committee on Standards of Official Conduct': ['Government Operations', 2000],
    'Committee on Transportation and Infrastructure': ['Transportation', 1000],
    'Committee on Commerce, Science, and Transportation': ['Domestic Commerce', 1500],
    'Committee on Science, Space, and Technology': ['Technology', 1700],
    'Committee on Commerce': ['Domestic Commerce', 1500],
    'Committee on the Budget': ['Macroeconomics', 100],
    'Committee on House Administration': ['Government Operations', 2000],
    'Committee on Oversight and Government Reform': ['Government Operations', 2000],
    'Special Committee on Aging': ['Social Welfare', 1300],
    'Select Committee on Intelligence': ['Defense', 1600],
    'Committee on Homeland Security': ['Defense', 1600],
    'Committee on Indian Affairs': ['Civil Rights', 200],
    'Committee on Rules and Administration': ['Government Operations', 2000],
    'Select Committee on Ethics': ['Government Operations', 2000],
    'Committee on Ethics': ['Government Operations', 2000],
    'Joint Select Committee on Deficit Reduction': ['Macroeconomics', 100],
    'Committee on Small Business': ['Domestic Commerce', 1500],
    'Temporary Joint Committee on Deficit Reduction': ['Macroeconomics', 100],
    'Joint Committee on Printing': ['Government Operations', 2000],
    'Select Committee on the Events Surrounding the 2012 Terrorist Attack in Benghazi': ['International Affairs', 1900],
    'Select Committee on Assassinations': ['Government Operations', 2000],
    'Committee on Education': ['Education', 600],
    'Committee on Banking and Currency': ['Domestic Commerce', 1500],
    'Committee on Oversight': ['Government Operations', 2000],
    'Joint Committee on the Library': ['Culture', 2300],
    'Committee on Public Works and Transportation': ['Transportation', 1000],
    'Committee on Interior and Insular Affairs': ['Public Lands', 2100],
    'Committee on Labor and Human Resources': ['Labor', 500],
    'Committee on Government Operations': ['Government Operations', 2000],
    'Committee on Science and Technology': ['Technology', 1700],
    'Committee on Governmental Affairs': ['Government Operations', 2000],
    'Select Committee on Energy Independence and Global Warming': ['Energy', 800],
    'Committee on Public Works': ['Public Lands', 2100],
    'Select Committee on Hunger': ['Social Welfare', 1300],
    'Select Committee on Presidential Campaign Activities': ['Government Operations', 2000],
    'Ad Hoc Committee on Energy': ['Energy', 800],
    'Committee on Merchant Marine and Fisheries': ['Domestic Commerce', 1500],
    'Committee on Public Lands': ['Public Lands', 2100],
    'Committee on Banking, Finance, and Urban Affairs': ['Domestic Commerce', 1500],
    'Joint Economic Committee': ['Macroeconomics', 100],
    'Committee on International Relations': ['International Affairs', 1900],
    'Committee on District of Columbia': ['Government Operations', 2000],
    'Select Committee on Standards and Conduct': ['Government Operations', 2000],
    'Committee on Science': ['Technology', 1700],
    'Committee on Agriculture and Forestry': ['Agriculture', 400],
    'Committee on Government Reform': ['Government Operations', 2000],
    'Joint Committee on Internal Revenue Taxation': ['Macroeconomics', 100],
    'Select Investigative Panel of the Committee on Energy and Commerce': ['Energy', 800],
    'Committee on Labor and Public Welfare': ['Social Welfare', 1300],
    'United States Senate Caucus on International Narcotics Control': ['International Affairs', 1900],
    'Committee on Internal Security': ['Law and Crime', 1200],
    'Joint Select Committee on Solvency of Multiemployer Pension Plans': ['Labor', 500],
    'Committee on National Security': ['Defense', 1600],
    'Select Committee on Aging': ['Social Welfare', 1300],
    'Joint Committee on the Organization of Congress': ['Government Operations', 2000]
    }

    # Map committees to their respective policy areas
    print("\nMapping committees to policy areas...")
    df_committees['policy_area'] = df_committees['committeeName'].map(committee_to_policy_area)

    # 5. Group by 'granuleId' and apply mode logic to determine dominant policy area
    print("\nCalculating dominant policy area for each 'granuleId'...")
    df_policy_areas = (
        df_committees.groupby('granuleId')['policy_area']
        .agg(calculate_mode)
        .reset_index()
    )

    # Split the 'policy_area' column into two separate columns for clarity
    print("\nSplitting policy area into separate columns...")
    df_policy_areas[['policy_area_name', 'policy_area_code']] = pd.DataFrame(
        df_policy_areas['policy_area'].tolist(), index=df_policy_areas.index
    )
    df_policy_areas.drop(columns=['policy_area'], inplace=True)

    # Display the processed DataFrame
    print("\nProcessed policy areas DataFrame:")
    print(df_policy_areas.head())

else:
    print("\nCommittees DataFrame is empty. Please check the file path or data integrity.")


