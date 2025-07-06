import paths
import pandas as pd
from scripts.preprocess.ExplorativeDataAnalysis import ExplorativeDataAnalysis
from scripts.preprocess.FeatureEngineering import FeatureEngineering
from scripts.unsupervised_learning.Clustering import Clustering


def load_data():
    df_x = []
    df_x0 = pd.read_json(paths.X0_DIR)
    df_x1 = pd.read_json(paths.X1_DIR)
    df_x2 = pd.read_json(paths.X2_DIR)

    df_x.append(df_x0)
    df_x.append(df_x1)
    df_x.append(df_x2)

    df_y = []
    df_y0 = pd.read_json(paths.Y0_DIR)
    df_y1 = pd.read_json(paths.Y1_DIR)
    df_y2 = pd.read_json(paths.Y2_DIR)

    df_y.append(df_y0)
    df_y.append(df_y1)
    df_y.append(df_y2)

    df_ds1 = []
    for p in paths.DS1_DIR:
        df = pd.read_json(p)
        df_ds1.append(df)

    # DS2 not a pandas datframe so do it later. it is also just  for puzzle

    return df_x, df_ds1, df_y


def dm_part1(df_x, df_ds1):
    eda_x = ExplorativeDataAnalysis(df_x)
    eda_x_df = eda_x.compute_eda("x_data_frame", plot=False,clean=True)

    eda_da1 = ExplorativeDataAnalysis(df_ds1)
    eda_da1_df = eda_da1.compute_eda("x_data_frame", plot=False,clean=True)

    feature_engineered_x = FeatureEngineering(eda_x_df)
    pca_x = feature_engineered_x.compute_features(show_plots=False)

    feature_engineered_ds1 = FeatureEngineering(eda_da1_df)
    pca_ds1 = feature_engineered_ds1.compute_features(show_plots=False)

    return eda_x_df, eda_da1_df


def dm_part2(df1, df2):
    kmean = Clustering(df1)
    dbscan = Clustering(df1)

    kmean.run_k_means(opt_k=True, input_k=True, subplots=False, evaluate=False)
    dbscan.run_DBSCAN(fast_compute=False)
    for i, df in enumerate(df1):
        print(f"\n=== Erweiterte Clusterverfahren für df{i} ===")
        kmean.fuzzy_c_means(df, c=3)
        kmean.mini_batch_kmeans(df, k=3)
        kmean.hierarchical_clustering(df, n_clusters=3)
        kmean.em_gaussian_mixture(df, n_components=3)


if __name__ == "__main__":
    df_x, df_ds1, df_y = load_data()

    df_x, df_ds1 = dm_part1( df_x, df_ds1)

    dm_part2(df_x, df_ds1)


