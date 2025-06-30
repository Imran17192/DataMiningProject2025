import Paths
import pandas as pd
from scripts.preprocess.ExplorativeDataAnalysis import ExplorativeDataAnalysis
from scripts.preprocess.FeatureEngineering import FeatureEngineering


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

    # DS2 not a pandas datframe so do it later. it is also just  for puzzle

    return df_x, df_ds1


def dm_part1():
    df_x, df_ds1 = load_data()

    eda_x = ExplorativeDataAnalysis(df_x)
    eda_x_df = eda_x.compute_eda("x_data_frame", plot=False,clean=True)

    eda_da1 = ExplorativeDataAnalysis(df_ds1)
    eda_da1_df = eda_da1.compute_eda("x_data_frame", plot=False,clean=True)

    feature_engineered_x = FeatureEngineering(eda_x_df)
    pca_x = feature_engineered_x.compute_features(show_plots=False)

    feature_engineered_ds1 = FeatureEngineering(eda_da1_df)
    pca_ds1 = feature_engineered_ds1.compute_features(show_plots=False)

    eda_x = ExplorativeDataAnalysis(pca_x)
    eda_x_df = eda_x.compute_eda("x_data_frame", plot=True)

    return pca_x, pca_ds1


def dm_part2(df1, df2):
   return 0


if __name__ == "__main__":

    df_x, df_ds1 = dm_part1()

    print("Hello World")