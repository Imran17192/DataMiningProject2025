import git
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

repo_path = git.Repo(__file__, search_parent_directories=True).working_dir
data_x_path = os.path.join(repo_path, "data", "x")
filenames = ["x0.json", "x1.json", "x2.json"]

dataframes = []

for filename in filenames:
	with open(os.path.join(data_x_path, filename)) as file:
		dataframes.append(pd.read_json(file))

def information(dataframe):
	print("-" * 100)
	print(f"Shape: {dataframe.shape}")
	print("-" * 50)
	print(f"Data types:\n{dataframe.dtypes}")
	print("-" * 50)
	print(f"Number of null values per column:\n{dataframe.isnull().sum()}")
	print("-" * 50)
	print(f"Descriptive statistics:\n{dataframe.describe}")
	print("-" * 50)
	print(f"Number of unique values per column:\n{dataframe.nunique()}")
	print("-" * 50)
	print(f"Number of duplicated rows: {dataframe.duplicated().sum()}")
	print("-" * 50)
	print(f"Head:\n{dataframe.head}")
	print("-" * 50)
	print(f"Tail:\n{dataframe.tail}")
	print("-" * 100)

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

def visualize(dataframe, title=None, type="heatmap"):
	plt.figure(figsize=(12, 10))
	if type == "heatmap":
		sns.heatmap(dataframe.corr(), annot=False, cmap='coolwarm', fmt='.2f', square=True)
		plt.title("Correlation" if not title else title)
	else:
		if len(dataframe.columns) > 2:
			pca = PCA(n_components=2)
			pca_result = pca.fit_transform(dataframe)
			plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c='blue')
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title("Visualization of main components" if not title else title)
		else:
			plt.scatter(dataframe[0], dataframe[1], color='blue', label='')
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title("Visualization" if not title else title)
	plt.show()

for dataframe in preprocessed_dataframes:
	visualize(dataframe)

for i, dataframe in enumerate(preprocessed_dataframes):
	visualize(dataframe, title=f"Scatter plot of x{i} after PCA", type="scatter")