import git
import json
import os

import numpy as np
import pandas as pd

repo_path = git.Repo(__file__, search_parent_directories=True).working_dir
data_x_path = os.path.join(repo_path, "data", "x")
filenames = ["x0.json", "x1.json", "x2.json"]

dataframes = []

for filename in filenames:
	with open(os.path.join(data_x_path, filename)) as file:
		dataframes.append(pd.read_json(file))

def information(dataframe):
    print(f"Shape: {dataframe.shape}")
    print()
    print(f"Data types:\n{dataframe.dtypes}")
    print()
    print(f"Number of null values per column:\n{dataframe.isnull().sum()}")
    print()
    print(f"Descriptive statistics:\n{dataframe.describe}")
    print()
    print(f"Number of unique values per column:\n{dataframe.nunique()}")
    print()
    print(f"Number of duplicated rows: {dataframe.duplicated().sum()}")
    print()
    print(f"Head:\n{dataframe.head}")
    print()
    print(f"Tail:\n{dataframe.tail}")

def remove_outliers(dataframe):
	Q1 = dataframe.quantile(1/4)
	Q3 = dataframe.quantile(3/4)

	IQR = Q3 - Q1

	lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return dataframe[~((dataframe < lower_bound) | (dataframe > upper_bound)).any(axis=1)]

def standardize(dataframe):
	return (dataframe - dataframe.mean()) / dataframe.std()

def normalize(dataframe):
	return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def center(dataframe):
	return dataframe - dataframe.mean()

for dataframe in dataframes:
	information(dataframe)
	print("-" * 100)

preprocessed_dataframes = []

for dataframe in dataframes:
	preprocessed_dataframe = center(standardize(remove_outliers(dataframe)))
	preprocessed_dataframes.append(preprocessed_dataframe)