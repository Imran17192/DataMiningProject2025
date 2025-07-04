import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Paths
from pandas.plotting import scatter_matrix, autocorrelation_plot
import math
import seaborn as sns

class ExplorativeDataAnalysis:
    def __init__(self, dfs):
        self.dfs = dfs

    def inspect_dataframe(self, dfs, name):
        print("------------------------------------------------------", name, "------------------------------------------------------")
        i = 1
        for df in dfs:
            print(i,"." , name, "dataframe\n")
            # first 5 lines of the dataframe
            print("head:")
            print(df.head())
            # last 5 lines of the dataframe
            print("tail:")
            print(df.tail())
            # class of the object, index range so how many line, column rang so how many columns
            # detailed descr of the columns, so amount, column name, how many lines aren't null, datatype of datapoint
            # shows memory usage
            print("info:")
            print(df.info())
            # gives us for each column count of lines, mean, std, min, ...
            print("describe:")
            print(df.describe())
            # checks for all columns if how many null values
            print("isNull.sum")
            print(df.isnull().sum())
            # tells the amount of unique values in each column
            print("nunique")
            print(df.nunique())
            i += 1


    def remove_outliers(self, dfs):
        df_cleaned = []
        for df in dfs:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df_no_outliers = df[((df >= lower_limit) & (df <= upper_limit)).all(axis=1)]
            df_cleaned.append(df_no_outliers)

        return df_cleaned


    def dfs_plot(self, dfs):
        for df in dfs:
            df.plot()

            plt.show()


    def standardize_df(self, dfs):
        dfs_std = []
        for df in dfs:
            df_std = (df - df.mean()) / df.std()
            dfs_std.append(df_std)
        return dfs_std


    def normalize_df(self, dfs):
        dfs_norm = []
        for df in dfs:
            df_norm = (df - df.min()) / (df.max() - df.min())
            dfs_norm.append(df_norm)
        return dfs_norm


    def plot_bar(self, dfs):
        for df in dfs:
            # hist creates plot for all columns of df columns and bins for amount of pillars
            df.hist(bins=100, figsize=(15, 10))
            plt.suptitle("Histogram plot of dataframe")
            plt.show()

    def plot_kernel(self, dfs):
        for df in dfs:
            D = df.shape[1]
            square_root = math.ceil(D ** 0.5)
            fig, axes = plt.subplots(nrows=square_root, ncols=square_root, figsize=(15, 10), constrained_layout=True)
            # flatten axes for easier iteration
            axes = axes.flatten()
            for i in range(D):
                sns.kdeplot(df.iloc[:, i], ax=axes[i])
                axes[i].set_title(f'Distribution of {df.columns[i]}')
            plt.show()

    def plot_heat(self, dfs):
        for df in dfs:
            corr = df.corr()
            plt.figure(figsize=(15, 10))
            sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linecolor="white")
            plt.show()

    def plot_index_scatter(self, dfs):
        for df in dfs:
            D = df.shape[1]
            square_root = math.ceil(D ** 0.5)
            fig, axes = plt.subplots(nrows=square_root, ncols=square_root,
                                     figsize=(15, 10), constrained_layout=True)
            axes = axes.flatten()
            for i in range(D):
                axes[i].scatter(df.index, df.iloc[:, i], alpha=0.5,
                                 marker='x', s=10)
                axes[i].set_title(f'Feature {df.columns[i]}')
                axes[i].set_xlabel('Index')
                axes[i].set_ylabel('Value')
            plt.show()

    def compute_eda(self, name, plot = False, clean=False):
        df = self.dfs

        if clean:
            df = self.standardize_df(df)

        self.inspect_dataframe(df, name)

        if plot:
            self.plot_bar(df)
            self.plot_kernel(df)
            self.plot_heat(df)
            self.plot_index_scatter(df)

        return df
    #-------------------------------------------------------------------
