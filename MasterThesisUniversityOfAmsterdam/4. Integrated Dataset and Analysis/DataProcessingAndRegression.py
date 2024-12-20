import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)

# **Function to Load Data**
def load_dataframe(file_path, file_type='csv'):
    """
    Load data from a CSV or JSON file.

    Parameters:
    - file_path (str): Path to the file.
    - file_type (str): File type ('csv' or 'json'). Defaults to 'csv'.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    if file_type == 'csv':
        return pd.read_csv(file_path, encoding='utf-8')
    elif file_type == 'json':
        return pd.read_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# **Function to Clean Data**
def clean_data(df):
    """
    Clean the dataset by handling missing values and anomalies.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Replace placeholder 'NA  ' with proper NaN
    df.replace('NA  ', np.nan, inplace=True)

    # Remove rows where 'org_id' exists but 'uuid_paragraph' is missing
    original_size = df.shape[0]
    invalid_rows = df[(df['org_id'].notna()) & (df['uuid_paragraph'].isna())]
    df = df[~((df['org_id'].notna()) & (df['uuid_paragraph'].isna()))]

    assert invalid_rows.shape[0] + df.shape[0] == original_size, "Row mismatch after filtering."
    return df

# **Function to Process Date Columns**
def process_date_columns(df):
    """
    Convert and compute new features from date-related columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Updated DataFrame with processed dates.
    """
    df['dateIssued_x'] = pd.to_datetime(df['dateIssued_x'], format='%Y-%m-%d', errors='coerce').dt.year
    df['FOUNDED'] = pd.to_numeric(df['FOUNDED'], errors='coerce')

    # Calculate how long organizations have existed
    df['YEARS_EXISTED'] = df['dateIssued_x'] - df['FOUNDED']
    df.loc[df['YEARS_EXISTED'] < 0, 'YEARS_EXISTED'] = np.nan
    df.loc[df['FOUNDED'] == 1900, 'YEARS_EXISTED'] = np.nan

    return df

# **Function to Assign Data Types**
def assign_data_types(df):
    """
    Assign appropriate data types to columns for optimized analysis.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Updated DataFrame with corrected data types.
    """
    type_mapping = {
        'prominence': float,
        'CATEGORY': 'category',
        'LOCATION': 'category',
        'MSHIP_STATUS11': 'category',
        'IN_HOUSE11': float,
        'OUTSIDE11': float,
        'LOBBYING11': float,
        'INHOUSEDUM11': 'category',
        'OUTSIDEDUM11': 'category',
        'LOBBYDUM11': 'category',
        'APPENDCAT': 'category',
        'ABBREVCAT': 'category',
        'partyHistory': 'category',
        'memberType': 'category',
        'stateCode': 'category',
        'stateName': 'category',
        'termBeginYear': float,
        'termEndYear': float,
        'PolicyAreas': 'category',
        'seniority': float,
        'issue_area_salience': float,
    }
    for column, dtype in type_mapping.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

# **Function to Normalize Salience**
def normalize_salience(df, column):
    """
    Normalize a salience column to represent proportions of the total salience.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column name to normalize.

    Returns:
    - pd.Series: Normalized salience as a proportion of the total.
    """
    return df[column] / df[column].sum()

# **Function to Create Ordinal Variables**
def create_ordinal_variable(df, column, bins, labels):
    """
    Create an ordinal variable by binning a numeric column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column name to convert to ordinal.
    - bins (int or list): Number of bins or list of bin edges.
    - labels (list): Labels for ordinal categories.

    Returns:
    - pd.Series: Ordinal variable as a new column.
    """
    return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

# **Function to Calculate Percentage Change**
def calculate_percentage_change(df, column):
    """
    Calculate the percentage change for a given column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column name to calculate percentage change for.

    Returns:
    - pd.Series: Percentage change for the column.
    """
    pct_change = df[column].pct_change()
    pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    return pct_change

# **Function to Add Lagged Variables**
def add_lagged_variables(df, column_list, lags=1):
    """
    Add lagged versions of specified columns to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_list (list): List of column names to create lagged variables for.
    - lags (int): Number of lag periods. Defaults to 1.

    Returns:
    - pd.DataFrame: DataFrame with added lagged columns.
    """
    for column in column_list:
        for lag in range(1, lags + 1):
            lagged_column_name = f"{column}_lag{lag}"
            df[lagged_column_name] = df[column].shift(lag)
    return df

# **Function to Visualize and Transform**
def transform_and_plot(df, column):
    """
    Apply a log transformation to a column and visualize the changes.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column name to transform.
    """
    # Replace zero values to avoid issues with log transformation
    df[column].replace(0, 0.001, inplace=True)

    # Original data visualizations
    plt.figure(figsize=(7, 6))
    sns.histplot(df[column], bins=30, kde=False)
    plt.title(f'Original Histogram: {column}')
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Original Boxplot: {column}')
    plt.show()

    # Apply log transformation
    df[f'{column}_log'] = np.log(df[column])

    # Transformed data visualizations
    plt.figure(figsize=(7, 6))
    sns.histplot(df[f'{column}_log'], bins=30, kde=False)
    plt.title(f'Log-Transformed Histogram: {column}')
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.boxplot(y=df[f'{column}_log'])
    plt.title(f'Log-Transformed Boxplot: {column}')
    plt.show()

# **Main Workflow**
if __name__ == "__main__":
    # File path to data
    file_path = "path_to_your_file.csv"  # Update with the actual path

    # Step 1: Load Data
    df = load_dataframe(file_path)

    # Step 2: Clean Data
    df = clean_data(df)

    # Step 3: Process Date Columns
    df = process_date_columns(df)

    # Step 4: Assign Data Types
    df = assign_data_types(df)

    # Step 5: Normalize Issue Area Salience
    df['issue_area_salience_recoded'] = normalize_salience(df, 'issue_area_salience')

    # Step 6: Create Ordinal and Percentage Change Variables
    df['issue_area_salience_ordinal'] = create_ordinal_variable(
        df, 'issue_area_salience_recoded', bins=21, labels=list(range(1, 22))
    )
    df['issue_area_salience_ordinal_3'] = create_ordinal_variable(
        df, 'issue_area_salience_recoded', bins=3, labels=[1, 2, 3]
    )
    df['issue_area_salience_pct_change'] = calculate_percentage_change(df, 'issue_area_salience_recoded')

    # Step 7: Add Lagged Variables
    lag_columns = [
        'issue_area_salience_recoded', 'issue_area_salience_ordinal', 
        'issue_area_salience_ordinal_3', 'issue_area_salience_pct_change'
    ]
    df = add_lagged_variables(df, lag_columns)

    # Step 8: Transform and Plot Variables
    for column in ['LOBBYING11', 'IN_HOUSE11', 'OUTSIDE11']:
        transform_and_plot(df, column)

    # Step 9: Save the Processed DataFrame
    output_path = "processed_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed DataFrame saved to {output_path}.")



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

# Configure pandas to display all columns for easier debugging
pd.set_option('display.max_columns', None)

# **Function to Standardize Data**
def standardize_data(df, columns_to_scale):
    """
    Standardize numeric columns to have zero mean and unit variance.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns_to_scale (list): List of column names to standardize.

    Returns:
    - pd.DataFrame: DataFrame with standardized columns.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df.index)
    df.update(scaled_df)  # Update the original DataFrame with standardized values
    return df

# **Function to Fit Probit Model**
def fit_probit_model(df, X_columns, y_column):
    """
    Fit a Probit regression model to predict a binary outcome.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - X_columns (list): Names of independent variable columns.
    - y_column (str): Name of the dependent variable column.

    Returns:
    - statsmodels object: Fitted Probit model.
    """
    # Drop rows with missing values in the specified columns
    df_clean = df.dropna(subset=X_columns + [y_column])
    X = sm.add_constant(df_clean[X_columns])  # Add a constant term for the regression
    y = df_clean[y_column]

    # Fit the Probit model
    probit_model = sm.Probit(y, X)
    probit_result = probit_model.fit()
    print(probit_result.summary())
    return probit_result

# **Function to Aggregate Data**
def aggregate_data(df, group_by_column, aggregation_rules):
    """
    Aggregate data based on specified rules for each group.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - group_by_column (str): Column to group by.
    - aggregation_rules (dict): Rules for aggregating each column.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with renamed columns for clarity.
    """
    aggregated_df = df.groupby(group_by_column).agg(aggregation_rules).reset_index()
    aggregated_df.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in aggregated_df.columns
    ]
    return aggregated_df

# **Function to Transform and Visualize Data**
def transform_and_plot(df, columns, transform_type='log', plot=True):
    """
    Apply transformations to columns and visualize the results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to transform.
    - transform_type (str): Type of transformation ('log', 'sqrt', 'boxcox').
    - plot (bool): Whether to generate visualizations.

    Returns:
    - pd.DataFrame: Updated DataFrame with transformed columns.
    """
    for column in columns:
        if transform_type == 'log':
            df[column].replace(0, 0.001, inplace=True)  # Handle zeros
            df[f"{column}_log"] = np.log(df[column])
        elif transform_type == 'sqrt':
            df[f"{column}_sqrt"] = np.sqrt(df[column])
        elif transform_type == 'boxcox':
            df[column].replace(0, 0.001, inplace=True)  # Handle zeros
            df[f"{column}_boxcox"], _ = stats.boxcox(df[column])

        if plot:
            # Plot original and transformed distributions
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df[column], bins=30, ax=axes[0])
            axes[0].set_title(f"Original Distribution: {column}")
            sns.histplot(df[f"{column}_{transform_type}"], bins=30, ax=axes[1])
            axes[1].set_title(f"{transform_type.capitalize()} Transformed: {column}")
            plt.show()

    return df

# **Main Workflow**
if __name__ == "__main__":
    # Step 1: Load the dataset
    file_path = "path_to_your_file.csv"  # Update with the actual file path
    df = pd.read_csv(file_path)

    # Step 2: Standardize numeric columns
    columns_to_standardize = ['LOBBYING11', 'IN_HOUSE11', 'OUTSIDE11']
    df = standardize_data(df, columns_to_standardize)

    # Step 3: Fit a Probit regression model
    X_columns = ['issue_maximal_overlap', 'partyHistory_Republican', 'partyHistory_Independent', 'congress_115.0']
    y_column = 'prominence'
    probit_result = fit_probit_model(df, X_columns, y_column)

    # Step 4: Aggregate data by organization ID
    aggregation_rules = {
        'prominence': 'sum',
        'paragraph_mention_count': 'sum',
        'YEARS_EXISTED': ['mean', 'median'],
        'LOBBYING11': ['mean', 'sum', 'median'],
        'IN_HOUSE11': ['mean', 'sum', 'median'],
        'OUTSIDE11': ['mean', 'sum', 'median'],
        'issue_area_salience': ['mean', 'median', 'sum'],
        'issue_area_salience_pct_change': 'mean',
    }
    aggregated_df = aggregate_data(df, 'org_id', aggregation_rules)

    # Step 5: Apply log transformation and visualize selected columns
    columns_to_transform = ['prominence', 'issue_area_salience', 'LOBBYING11']
    transformed_df = transform_and_plot(aggregated_df, columns_to_transform, transform_type='log')

    # Step 6: Save the processed DataFrame to a CSV file
    output_path = "processed_data.csv"
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated data saved to {output_path}.")
