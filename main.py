import paths
import pandas as pd

from scripts.preprocess.ExplorativeDataAnalysis import ExplorativeDataAnalysis
from scripts.preprocess.FeatureEngineering import FeatureEngineering
from scripts.preprocess.Preprocessing import Preprocessing
from scripts.train.HierarchicalClustering import HierarchicalClustering
from scripts.train.kMeans import kMeans
from scripts.unsupervised_learning.Clustering import Clustering
from scripts.visualization.Visualization import Visualization


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
    hierarchical_clustering = Clustering(df1)
    kmean = Clustering(df1)
    dbscan = Clustering(df1)

    for linkage_method in ["single", "complete", "average"]:
        hierarchical_clustering.silhouette_analysis(linkage_method)
        hierarchical_clustering.linkage_clustering(linkage_method)

    hierarchical_clustering_results = hierarchical_clustering.get_clustering_results()
    for clustering_method in hierarchical_clustering_results:
        for dataframe in hierarchical_clustering_results[clustering_method]:
            Visualization.visualize_clusters(
                hierarchical_clustering_results[clustering_method][dataframe]["dataframe"],
                hierarchical_clustering_results[clustering_method][dataframe]["labels"],
                title=f"Clusteringergebnis {clustering_method}-Linkage ({dataframe}"
            )

    kmean.run_k_means(opt_k=True, input_k=True, subplots=False, evaluate=False)
    dbscan.run_DBSCAN(fast_compute=False)


if __name__ == "__main__":
    df_x, df_ds1, df_y = load_data()

    df_x, df_ds1 = dm_part1( df_x, df_ds1)

    dm_part2(df_x, df_ds1)