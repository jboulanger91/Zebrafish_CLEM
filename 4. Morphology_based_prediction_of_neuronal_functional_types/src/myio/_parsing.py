"""Internal parsing utilities for myio module.

This module contains shared parsing functions used by multiple loaders.
These are internal utilities (prefixed with _) and should not be imported directly.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def read_to_pandas_row(file_content: str) -> pd.DataFrame:
    r"""Parse key=value text format into a single-row DataFrame.

    Parses a text file where each line contains a key=value pair,
    handling special cases for lists, NaN values, and numeric strings.

    Parameters
    ----------
    file_content : str
        Text content with lines in "key=value" format.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with parsed key-value pairs as columns.

    Examples
    --------
    >>> df = read_to_pandas_row("name=cell_1\ncount=42\nvalues=[1, 2, 3]")
    >>> df['name'].iloc[0]
    'cell_1'
    >>> int(df['count'].iloc[0])
    42
    >>> df['values'].iloc[0]
    [1, 2, 3]
    """
    data_list = file_content.strip().split('\n')

    # Parse the list to create a dictionary
    data_dict = {}
    for item in data_list:
        key, value = item.split("=", 1)  # Split by the first "=" only
        key = key.strip()
        value = value.strip().strip('"')

        # Special handling for list and nan values
        if value.startswith("[") and value.endswith("]"):
            # Convert string list to actual list (ast.literal_eval is safe)
            value = ast.literal_eval(value)
        elif value == "nan":
            value = np.nan
        elif value.isdigit():
            # Convert numeric strings to integers
            value = int(value)

        data_dict[key] = value

    # Create a DataFrame row
    df_row = pd.DataFrame([data_dict])
    return df_row
