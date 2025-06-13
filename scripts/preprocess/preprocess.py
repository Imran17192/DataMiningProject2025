import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Paths
from pandas.plotting import scatter_matrix, autocorrelation_plot
import math
import seaborn as sns



def load_data():
    df_x = []
    df_x0 = pd.read_json(Paths.X0_DIR)
    df_x1 = pd.read_json(Paths.X1_DIR)
    df_x2 = pd.read_json(Paths.X2_DIR)
    df_x.append(df_x0)
    df_x.append(df_x1)
    df_x.append(df_x2)

    df_ds1 = []
    for p in Paths.DS1_DIR:
        df = pd.read_json(p)
        df_ds1.append(df)

    # TODO DS2 not a pandas datframe so do it later

    return df_x, df_ds1


def inspect_dataframe(dfs, name):
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
        # detailed descr of the colums, so amount, column name, how many lines arent null, datatype of datapoint
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


def remove_oultiers(df_x):
    df_cleaned = []
    for df in df_x:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        df_no_outliers = df[((df >= lower_limit) & (df <= upper_limit)).all(axis=1)]
        df_cleaned.append(df_no_outliers)

    return df_cleaned


def dfs_plot(dfs):
    for df in dfs:
        df.plot()

        plt.show()


def standardize_df(dfs):
    dfs_std = []
    for df in dfs:
        df_std = (df - df.mean()) / df.std()
        dfs_std.append(df_std)
    return dfs_std


def normalize_df(dfs):
    dfs_norm = []
    for df in dfs:
        df_norm = (df - df.min()) / (df.max() - df.min())
        dfs_norm.append(df_norm)
    return dfs_norm


def plot_bar(dfs):
    for df in dfs:
        # hist creates plot for all columns of df columns and bins for amount of pillars
        df.hist(bins=100, figsize=(15, 10))
        plt.suptitle("Histogram plot of x dataframe")
        plt.show()

def plot_kernel(dfs):
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

def plot_heat(dfs):
    for df in dfs:
        corr = df.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linecolor="white",
            cbar_kws={"shrink": .8}
        )
        plt.show()


if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    inspect_dataframe(df_x, "x_data_frame")
    df_x_clean = remove_oultiers(df_x)

    df_x_stand = standardize_df(df_x_clean)
    df_x_norm = normalize_df(df_x_clean)

    plot_bar(df_x_stand)
    plot_kernel(df_x_stand)
    plot_heat(df_x_stand)

    inspect_dataframe(df_ds1, "df_ds1")
    df_ds1_clean = remove_oultiers(df_ds1)

    df_ds1_stand = standardize_df(df_ds1_clean)
    df_ds1_norm = normalize_df(df_ds1_clean)

    plot_bar(df_ds1_stand)
    plot_kernel(df_ds1_stand)
    plot_heat(df_ds1_stand)

