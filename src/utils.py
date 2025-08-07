#!/usr/bin/env python
# coding: utf-8

# In[1]:


def print_structure_report(data, remove=None, change=None):
    """
    Generates and prints a structured report summarizing a pandas DataFrame.

    This function provides a detailed overview of the dataset, including its shape,
    missing values, duplicate entries, and a breakdown of features by data type
    (categorical, numerical, and boolean). For each feature, it lists the count
    of unique values and a sample of those values or a min/max range for numerical
    data. It concludes with a "Next Steps" section to guide further data
    preprocessing.

    Args:
        data (pd.DataFrame): The input DataFrame to be analyzed.
        remove (list, optional): A list of column names to be suggested for removal
                                 in the "Next Steps" section. Defaults to None.
        change (list, optional): A list of column names to be suggested for a
                                 data type change in the "Next Steps" section.
                                 Defaults to None.

    Returns:
        None. The function prints the report directly to the console.
    """
    # Initialize lists if they are None to prevent errors
    if remove is None:
        remove = []
    if change is None:
        change = []

    tab = '   '
    # Define the header for the feature tables
    head = (f'{"Features":<25} {"Unique Values Count":<25} {"Unique Values":<25}\n'
            f'{"-"*25:<25} {"-"*25:<25} {"-"*98:<25}')

    print(f'{21 * tab}Data Structure Report')
    print(150 * '=')

    # --- Data Profiling ---
    # Identify features by data type
    cat = data.select_dtypes(include='object').columns.to_list()
    num = data.select_dtypes(include=['number']).columns.to_list()
    bol = data.select_dtypes(include=['bool']).columns.to_list()

    # Calculate missing values and find columns with NaNs
    nan_count = data.isna().sum()
    nan_cols = nan_count[nan_count > 0].index.tolist()

    # Count duplicated rows
    dup_count = data.duplicated().sum()

    # --- Print Summary Section ---
    print(f'Summary:')
    print(f'• The dataset has {data.shape[0]} observations and {data.shape[1]} features.')
    if nan_count.sum() == 0:
      print(f'• There are no missing values in the dataset.')
    else:
      print(f'• There are {nan_count.sum()} missing values in {nan_cols}.')
    if dup_count == 0:
      print(f'• There are no duplicated values in the dataset.')
    else:
      print(f'• There are {dup_count} duplicated values in the dataset.')
    print(f'• There are {len(cat)} categorical features, {len(num)} numerical features, and {len(bol)} boolean features.\n')

    # --- Print Categorical Features Details ---
    if len(cat) > 0:
      print(f'Categorical Features: {len(cat)}')
      print(head)
      for col in cat:
          unique_count = data[col].nunique()
          # To avoid printing too many values, show a message instead
          if unique_count > 10:
              unique_values = "Too many values"
          else:
              unique_values = ', '.join(map(str, data[col].unique().tolist()))
          print(f'{col:<25} {unique_count:<25} {unique_values:<25}')

    # --- Print Numerical Features Details ---
    if len(num) > 0:
      print(f'\nNumerical Features: {len(num)}')
      print(head)
      for col in num:
        unique_count = data[col].nunique()
        min_value = data[col].min()
        max_value = data[col].max()
        # For numerical columns, show the min and max values
        unique_values = f"Min: {min_value:<14.2f} Max: {max_value:.2f}"
        print(f'{col:<25} {unique_count:<25} {unique_values:<25}')

    # --- Print Boolean Features Details ---
    if len(bol) > 0:
      print(f'\nBoolean Features: {len(bol)}')
      print(head)
      for col in bol:
          unique_count = data[col].nunique()
          unique_values = ', '.join(map(str, data[col].unique().tolist()))
          print(f'{col:<25} {unique_count:<25} {unique_values:<25}')


    print(150 * '=')

    # --- Print Next Steps Section ---
    print('\nNext Steps:')
    if dup_count > 0:
      print(f'• Remove duplicated values.')
    if len(remove) > 0:    
        print(f'• Remove the features: {remove}')
    if len(change) > 0:
        print(f'• Change the data type of the features: {change}')
    # Generic suggestions for any data science project
    print('• Standardize feature names.')
    print('• Split the data into train and test sets.')
    print('• Perform Exploratory Data Analysis (EDA).')

