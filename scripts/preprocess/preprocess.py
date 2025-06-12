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


def data_fram_information(dfs):
  return 0

if __name__ == "__main__":
    df_x, df_ds1 = load_data()
    data_fram_information(df_x)
    data_fram_information(df_ds1)

