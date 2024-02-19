import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
import warnings
from typing import List, Tuple, Optional, Union, Literal
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
import sqlite3

@contextmanager
def timer(title: str):
    """
    A context manager that measures the time taken to execute a block of code.

    Args:
        title (str): The title of the code block.

    Yields:
        None

    Prints:
        The title of the code block and the time taken to execute it.

    Example:
        >>> with timer("Function execution"):
        ...     # code block
        Function execution - done in 5s
    """
    start_time = time.time()
    yield
    print(f"{title} - done in {time.time() - start_time:.0f}s")

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encodes the categorical columns in the given DataFrame.

    Args:
        df: The input DataFrame to be encoded.
        nan_as_category: Whether to treat NaN values as a separate category. Defaults to True.

    Returns:
        A tuple containing the encoded DataFrame and a list of new columns.
    """
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = list(set(encoded_df.columns) - set(original_columns))
    return encoded_df, new_columns

# Preprocess application_train.csv 
def application_train_dataset(num_rows=None, nan_as_category=False):
    """
    Reads the application train and test data from CSV files and merges them into a single DataFrame.

    Args:
        num_rows (int, optional): The number of rows to read from the CSV files. Defaults to None.
        nan_as_category (bool, optional): Whether to treat NaN values as a category. Defaults to False.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the train and test data.
    """
    # Read data and merge
    train_data = pd.read_csv('./input/application_train.csv', nrows=num_rows, sep=',', encoding='utf_8').reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    train_data = train_data[train_data['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    binary_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for feature in binary_features:
        train_data[feature], _ = pd.factorize(train_data[feature])

    # Categorical features with One-Hot encode
    train_data, _ = one_hot_encoder(train_data, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    train_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Some simple new features (percentages)
    train_data['DAYS_EMPLOYED_PERC'] = train_data['DAYS_EMPLOYED'] / train_data['DAYS_BIRTH']
    train_data['INCOME_CREDIT_PERC'] = train_data['AMT_INCOME_TOTAL'] / train_data['AMT_CREDIT']
    train_data['INCOME_PER_PERSON'] = train_data['AMT_INCOME_TOTAL'] / train_data['CNT_FAM_MEMBERS']
    train_data['ANNUITY_INCOME_PERC'] = train_data['AMT_ANNUITY'] / train_data['AMT_INCOME_TOTAL']
    train_data['PAYMENT_RATE'] = train_data['AMT_ANNUITY'] / train_data['AMT_CREDIT']

    train_data.set_index('SK_ID_CURR', inplace=True)

    return train_data

# Preprocess application_train.csv 
def application_test_dataset(num_rows=None, nan_as_category=False):
    """
    Reads the application train and test data from CSV files and merges them into a single DataFrame.

    Args:
        num_rows (int, optional): The number of rows to read from the CSV files. Defaults to None.
        nan_as_category (bool, optional): Whether to treat NaN values as a category. Defaults to False.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the train and test data.
    """
    # Read data and merge
    test_data = pd.read_csv('./input/application_test.csv', nrows=num_rows, sep=',', encoding='utf_8').reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    test_data = test_data[test_data['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    binary_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for feature in binary_features:
        test_data[feature], _ = pd.factorize(test_data[feature])

    # Categorical features with One-Hot encode
    test_data, _ = one_hot_encoder(test_data, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    test_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Some simple new features (percentages)
    test_data['DAYS_EMPLOYED_PERC'] = test_data['DAYS_EMPLOYED'] / test_data['DAYS_BIRTH']
    test_data['INCOME_CREDIT_PERC'] = test_data['AMT_INCOME_TOTAL'] / test_data['AMT_CREDIT']
    test_data['INCOME_PER_PERSON'] = test_data['AMT_INCOME_TOTAL'] / test_data['CNT_FAM_MEMBERS']
    test_data['ANNUITY_INCOME_PERC'] = test_data['AMT_ANNUITY'] / test_data['AMT_INCOME_TOTAL']
    test_data['PAYMENT_RATE'] = test_data['AMT_ANNUITY'] / test_data['AMT_CREDIT']

    test_data.set_index('SK_ID_CURR', inplace=True)

    return test_data

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows: Union[int, None] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Read the 'bureau.csv' and 'bureau_balance.csv' files and perform data 
    preprocessing and feature engineering on them. 

    Parameters:
    - num_rows (int, optional): Number of rows to read from the CSV files.
      Defaults to None, which reads the entire files.
    - nan_as_category (bool, optional): If True, treat NaN values as a 
      separate category. Defaults to True.

    Returns:
    - bureau_agg (pd.DataFrame): The aggregated and processed data from the 
      'bureau.csv' and 'bureau_balance.csv' files grouped by 'SK_ID_CURR'.
    """
    # Read bureau.csv and bureau_balance.csv files
    bureau = pd.read_csv('./input/bureau.csv', nrows=num_rows, sep=',')
    bureau_balance = pd.read_csv('./input/bureau_balance.csv', nrows=num_rows, sep=',')
    
    # Perform one-hot encoding on bureau_balance and bureau datasets
    bureau_balance, bureau_balance_cat = one_hot_encoder(bureau_balance, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Define aggregations for bureau_balance
    bureau_balance_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bureau_balance_cat:
        bureau_balance_aggregations[col] = ['mean']
    
    # Group by 'SK_ID_BUREAU' and aggregate bureau_balance data
    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_aggregations)
    bureau_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance_agg.columns.tolist()])
    
    # Join bureau and bureau_balance_agg datasets and drop unnecessary columns
    bureau = bureau.join(bureau_balance_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    
    # Clean up memory
    del bureau_balance, bureau_balance_agg
    
    # Define aggregations for numerical columns and categorical columns
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bureau_balance_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']
    
    # Group by 'SK_ID_CURR' and aggregate numerical and categorical columns
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau

    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows: Union[int, None] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Reads the 'previous_application.csv' file and performs data preprocessing and feature engineering on it.
    
    Parameters:
    - num_rows: int, optional
        The number of rows to read from the file. If not specified, all rows are read.
    - nan_as_category: bool, optional
        Flag indicating whether to treat NaN values as a separate category. Defaults to True.
    
    Returns:
    - prev_agg: pandas DataFrame
        The aggregated and processed data from the 'previous_application.csv' file.
    """
    prev = pd.read_csv('./input/previous_application.csv', nrows=num_rows, sep=',')
    
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    cat_aggregations = {cat: ['mean'] for cat in cat_cols}
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index([f'PREV_{e[0]}_{e[1].upper()}' for e in prev_agg.columns.tolist()])
    
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index([f'APPROVED_{e[0]}_{e[1].upper()}' for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    return prev_agg
   

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows: Optional[int] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Reads the 'POS_CASH_balance.csv' file and performs data preprocessing and feature engineering on it.
    
    Parameters:
        num_rows (int, optional): The number of rows to read from the file. If not specified, reads the entire file.
        nan_as_category (bool, optional): Whether to treat NaN values as a separate category. Defaults to True.
        
    Returns:
        pos_agg (DataFrame): The aggregated and processed data from the 'POS_CASH_balance.csv' file.
    """
    # Read the 'POS_CASH_balance.csv' file
    pos = pd.read_csv('./input/POS_CASH_balance.csv', nrows=num_rows, sep=',')
    
    # Perform one-hot encoding on the categorical columns
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)
    
    # Define aggregations
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    # Add mean aggregation for each categorical column
    for cat_col in cat_cols:
        aggregations[cat_col] = ['mean']
    
    # Aggregate data by grouping it by 'SK_ID_CURR' and applying the defined aggregations
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    
    # Rename the columns of the aggregated data
    pos_agg.columns = pd.Index(['POS_' + col[0] + "_" + col[1].upper() for col in pos_agg.columns.tolist()])
    
    # Count the number of pos cash accounts for each 'SK_ID_CURR' and add it as a new column
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    # Clean up memory
    del pos
    
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows: Optional[int] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Reads the 'installments_payments.csv' file and performs various transformations and aggregations on the data.
    
    Args:
        num_rows (int, optional): The number of rows to read from the file. Defaults to None.
        nan_as_category (bool, optional): Whether to treat NaN values as a separate category. Defaults to True.
        
    Returns:
        pd.DataFrame: The aggregated and transformed data with various features.
    """
    # Read 'installments_payments.csv' file
    ins = pd.read_csv('./input/installments_payments.csv', nrows=num_rows, sep=',')
    
    # Apply one-hot encoding to categorical columns
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)
    
    # Calculate percentage and difference paid in each installment
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Calculate days past due and days before due
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    # Clean up memory
    del ins
    
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(max_rows: Optional[int] = None, nan_as_category: bool = True) -> pd.DataFrame:
    """
    Reads credit card balance data from a CSV file and performs aggregation on it.

    Args:
        max_rows (int, optional): Maximum number of rows to read from the CSV file. Default is None.

    Returns:
        pd.DataFrame: Aggregated credit card balance data.
    """
    # Read credit card balance data from CSV file
    credit_card_balance_df = pd.read_csv('./input/credit_card_balance.csv', nrows=max_rows, sep=',')

    credit_card_balance_df, _ = one_hot_encoder(credit_card_balance_df, nan_as_category)

    credit_card_balance_df.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    
    # Perform aggregation on credit card balance data
    credit_card_balance_agg = credit_card_balance_df.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    
    # Rename columns with 'CC_' prefix
    credit_card_balance_agg.columns = ['CC_' + '_'.join(col).upper() for col in credit_card_balance_agg.columns]
    
    # Add 'CC_COUNT' column with count of rows per 'SK_ID_CURR'
    credit_card_balance_agg['CC_COUNT'] = credit_card_balance_df.groupby('SK_ID_CURR').size()
    
    return credit_card_balance_agg

def select_common_columns(df1: pd.DataFrame, df2: pd.DataFrame, output: Literal['df1', 'df2', 'both']) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Select common columns from two dataframes.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        pd.DataFrame or tuple: The dataframe 1 or 2 or both with only common columns only.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    if output == 'df1':
        return df1
    elif output == 'df2':
        return df2
    else:
        return df1, df2

def data_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a pandas DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test sets.
    """

    train_index = train_test_split(df, test_size=0.3, random_state=13, stratify=df['TARGET'])[0].index
    test_index = df.index.difference(train_index)

    train_df = df.loc[train_index]
    test_df = df.loc[test_index]

    return train_df, test_df

def nan_imputation(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None, inf_method: Union[str, None] = 'nan', nan_method: Union[str, None] = 'median') -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Imputes missing values in a pandas DataFrame using specified methods.

    Args:
        train_df (pd.DataFrame): The DataFrame containing training data, which will be used to implement nan in both train and test sets.
        test_df (pd.DataFrame): The DataFrame containing test data, which will be filled with imputed values from train set.
        inf_method (Union[str, None], optional): The method to use for replacing infinite values.
            Defaults to 'nan'.
        nan_method (Union[str, None], optional): The method to use for replacing NaN values.
            Defaults to 'median'.

    Returns:
        None
    """

    train_copy = train_df.copy()
    if test_df is not None:
        test_copy = test_df.copy()

    inf_methods = ['nan', 'mean', 'median', 'mode']
    nan_methods = ['mean', 'median', 'mode']

    if inf_method == 'nan':
        train_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        if test_df is not None:
            test_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    elif inf_method in inf_methods:
        for col in train_copy.columns:
            if inf_method == 'mean':
                fill_value = train_copy[col].mean()
            elif inf_method == 'median':
                fill_value = train_copy[col].median()
            elif inf_method == 'mode':
                fill_value = train_copy[col].mode().iloc[0]
            else:
                fill_value = np.nan
            train_copy[col].fillna(fill_value, inplace=True)
            if test_df is not None:
                test_copy[col].fillna(fill_value, inplace=True)
    
    if nan_method in nan_methods:
        for col in train_copy.columns:
            if nan_method == 'mean':
                fill_value = train_copy[col].mean()
            elif nan_method == 'median':
                fill_value = train_copy[col].median()
            elif nan_method == 'mode':
                fill_value = train_copy[col].mode().iloc[0]
            else:
                fill_value = np.nan
            train_copy[col].fillna(fill_value, inplace=True)
            if test_df is not None:
                test_copy[col].fillna(fill_value, inplace=True)

    train_copy.dropna(axis=1, how='all', inplace=True)
    if test_df is not None:
        test_copy.dropna(axis=1, how='all', inplace=True)

    if test_df is not None:
        train_df = train_copy
        test_df = test_copy
        return train_df, test_df
    else:
        train_df = train_copy
        return train_df

def clean_columns_names(df):
    column_names = df.columns.to_list()
    replace_dict = {
        ',': '_',
        ':': '_',
        '-': '_',
        '/': '_',
        '(': '',
        ')': '',
        '+': '',
        '"': '',
        ' ': '_',
        '__': '_',
        '__': '_'
    }

    for i in range(len(column_names)):
        for old_char, new_char in replace_dict.items():
            column_names[i] = column_names[i].replace(old_char, new_char)
    
    df.columns = column_names
    return df

def order_columns(df):
    """
    Order the columns names of a DataFrame and returns the reordered DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to extract the column names from.

    Returns:
        pandas.DataFrame: The original DataFrame with the reordered column names.
    """
    # Extract column list.
    columns_names = df.columns.to_list()
    columns_names.sort()
    if 'TARGET' in columns_names:
        for i, col in enumerate(['index', 'TARGET']):
            columns_names.insert(i, columns_names.pop(columns_names.index(col)))
    else:
        for i, col in enumerate(['index']):
            columns_names.insert(i, columns_names.pop(columns_names.index(col)))

    return df[columns_names]

def save_to_db(df, table_name, db_path='clients_infos'):
    """
    Saves a dataframe as a table in a SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to save as a table.
        table_name (str): The name of the table in the database.
        db_name (str): The name of the database.

    Returns:
        None
    """

    conn = sqlite3.connect(f'features/{db_path}.db')
    df.to_sql(f'{table_name}', conn, if_exists='replace', index=True)
    conn.close()

def main(debug: bool = True, split: bool = True, data_drift_report: bool = False):
    num_rows = 10000 if debug else None
    df = application_train_dataset(num_rows)
    df_test = application_test_dataset(num_rows)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.merge(bureau, how='left', on='SK_ID_CURR', validate='1:1')
        df_test = df_test.merge(bureau, how='left', on='SK_ID_CURR', validate='1:1')
        del bureau

    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.merge(prev, how='left', on='SK_ID_CURR', validate='1:1')
        df_test = df_test.merge(prev, how='left', on='SK_ID_CURR', validate='1:1')
        del prev

    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.merge(pos, how='left', on='SK_ID_CURR', validate='1:1')
        df_test = df_test.merge(pos, how='left', on='SK_ID_CURR', validate='1:1')
        del pos

    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.merge(ins, how='left', on='SK_ID_CURR', validate='1:1')
        df_test = df_test.merge(ins, how='left', on='SK_ID_CURR', validate='1:1')
        del ins
    
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.merge(cc, how='left', on='SK_ID_CURR', validate='1:1')
        df_test = df_test.merge(cc, how='left', on='SK_ID_CURR', validate='1:1')
        del cc
    
    with timer("Final encoding, data type changes"):
        df.drop(columns='index')
        df_test.drop(columns='index')
        # print(df.columns, df_test.columns)
        print("features_df shape:", df.shape)
        print("test_df shape:", df_test.shape)
        df, _ = one_hot_encoder(df, nan_as_category=True)
        df_test, _ = one_hot_encoder(df_test, nan_as_category=True)
        bool_columns = df.select_dtypes(include='bool').columns
        bool_columns_test = df_test.select_dtypes(include='bool').columns
        df[bool_columns] = df[bool_columns].astype('int8')
        df_test[bool_columns_test] = df_test[bool_columns_test].astype('int8')
        df = clean_columns_names(df)
        df_test = clean_columns_names(df_test)

    if data_drift_report:
        with timer("Data drift report"):
            full_train_df, full_test_df = df.drop(columns='TARGET'), df_test

        with timer("Selecting columns common to full train and full test"):
            print(full_train_df.shape, full_test_df.shape)
            full_train_df, full_test_df = select_common_columns(full_train_df, full_test_df, 'both')

        with timer("NaN imputation"):
            nan_values_nb_full_train = full_train_df.isna().sum().sum()
            nan_values_nb_full_test = full_test_df.isna().sum().sum()
            full_train_df, full_test_df = nan_imputation(full_train_df, full_test_df)
            print("NaN values imputed in full train set: ", nan_values_nb_full_train - full_train_df.isna().sum().sum())
            print("NaN values imputed in full test set: ", nan_values_nb_full_test - full_test_df.isna().sum().sum())

        with timer("Selecting columns common to full train and full test"):
            print(full_train_df.shape, full_test_df.shape)
            full_train_df, full_test_df = select_common_columns(full_train_df, full_test_df, 'both')

        with timer("Data saving"):
            filenames = ["full_train_df_debug", "full_test_df_debug"] if debug else ["full_train_df", "full_test_df"]
            print(filenames)
            for filename, dataframe in zip(filenames, [full_train_df, full_test_df]):
                with open(f'features/{filename}.csv', 'w') as file:
                    print(filename, dataframe.shape)
                    dataframe = order_columns(dataframe)
                    save_to_db(dataframe, table_name=filename)
                    dataframe.to_csv(file)
    else:
        if split:
            with timer("Data split"):
                train_df, test_df = data_split(df)
                print("Train df shape:", train_df.shape)
                print("Test df shape:", test_df.shape)

            with timer("NaN imputation"):
                nan_values_nb_train = train_df.isna().sum().sum()
                nan_values_nb_test = test_df.isna().sum().sum()
                train_df, test_df = nan_imputation(train_df, test_df)
                print("NaN values imputed in train set: ", nan_values_nb_train - train_df.isna().sum().sum())
                print("NaN values imputed in test set: ", nan_values_nb_test - test_df.isna().sum().sum())

            with timer("Selecting columns common to train and test"):
                print(train_df.shape, test_df.shape)
                train_df, test_df = select_common_columns(train_df, test_df, 'both')
            
            with timer("Data saving"):
                filenames = ["train_df_debug", "test_df_debug"] if debug else ["train_df", "test_df"]
                print(filenames)
                for filename, dataframe in zip(filenames, [train_df, test_df]):
                    with open(f'features/{filename}.csv', 'w') as file:
                        print(filename, dataframe.shape)
                        dataframe = order_columns(dataframe)
                        save_to_db(dataframe, table_name=filename)
                        dataframe.to_csv(file)            
                        
        else:
            with timer("NaN imputation"):
                nan_values_nb = df.isna().sum().sum()
                df = nan_imputation(df)
                print("NaN values imputed: ", nan_values_nb - df.isna().sum().sum())

            with timer("Data saving"):
                filename = 'features_df_debug' if debug else "features_df"
                with open (f'features/{filename}.csv', 'w') as file:
                    df = order_columns(df)
                    save_to_db(df, table_name=filename)
                    df.to_csv(file)
    


main(debug=False, split=False, data_drift_report=False)