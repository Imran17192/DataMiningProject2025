import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Paths

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


if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    inspect_dataframe(df_x, "x_data_frame")
    df_x_clean = remove_oultiers(df_x)
    inspect_dataframe(df_x_clean, "x_data_frame removed ouliers")

    inspect_dataframe(df_ds1, "ds_data_frame")
    df_ds1 = remove_oultiers(df_ds1)
    inspect_dataframe(df_ds1, "ds_data_frame removed ouliers")



    # TODO df shapes
    # TODO df cleaning
    # TODO df scaling ana normalizing
    # TODO df visualizations
    # TODO filter noise
    # TODO heatmaps
    # TODO PCA
    # TODO Quantifying
    # TODO Automated Branch and bound

