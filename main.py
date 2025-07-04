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
    kmean = Clustering(df1)
    dbscan = Clustering(df1)

    kmean.run_k_means(opt_k=True, input_k=True, subplots=False, evaluate=False)
    dbscan.run_DBSCAN(fast_compute=False)


if __name__ == "__main__":
    df_x, df_ds1, df_y = load_data()

    df_x, df_ds1 = dm_part1( df_x, df_ds1)

    dm_part2(df_x, df_ds1)

    # Preprocessing

    dataframes_preprocessed = []

    for dataframe in df_x:
        dataframes_preprocessed.append(Preprocessing.preprocess(dataframe))

    for i, dataframe_preprocessed in enumerate(dataframes_preprocessed):
        Visualization.visualize_dataframe(dataframe_preprocessed, title=f"Datensatz x{i} (vorverarbeitet)")

    # Clustering

    ## Hierarchisches Clustering

    hierarchical_clustering = HierarchicalClustering(dataframes_preprocessed)

    linkage_methods = ["single", "average", "complete"]

    clustering_results = {
        f"x{i}": {
            "dataframe": dataframe_preprocessed,
            "hierarchical_clustering": {
                linkage_method: {} for linkage_method in linkage_methods
            }
        } for i, dataframe_preprocessed in enumerate(dataframes_preprocessed)
    }

    for linkage_method in linkage_methods:
        dataframes_silhouette_scores = hierarchical_clustering.silhouette_analysis(linkage_method, c_max=30)
        for i, dataframe_silhouette_scores in enumerate(dataframes_silhouette_scores):
            Visualization.silhouette_analysis(dataframe_silhouette_scores,
                                              title=f"Silhouettenkoeffizienten für x{i} ({linkage_method}-Linkage-Verfahren)")
            c = max(dataframe_silhouette_scores, key=lambda x: x[1])[0]
            clustering_results[f"x{i}"]["hierarchical_clustering"][linkage_method]["c"] = c

    for x in clustering_results:
        dataframe = clustering_results[x]["dataframe"]
        for linkage_method in clustering_results[x]["hierarchical_clustering"]:
            c = clustering_results[x]["hierarchical_clustering"][linkage_method]["c"]
            labels = HierarchicalClustering.linkage_clustering(dataframe, linkage_method, c)
            clustering_results[x]["hierarchical_clustering"][linkage_method]["labels"] = labels
            Visualization.visualize_clusters(dataframe, labels,
                                             title=f"Clusteringergebnis für {x} mit c={c} ({linkage_method}-Linkage-Verfahren)")

    ## k-Means

    k_means = kMeans(dataframes_preprocessed)

    dataframes_wcss = k_means.elbow_analysis(k_max=30)
    for i, dataframe_wcss in enumerate(dataframes_wcss):
        Visualization.elbow_analysis(dataframe_wcss, title=f"Ellbogendiagramm für x{i} (k-Means)")
        k = kMeans.locate_knee(dataframe_wcss)
        clustering_results[f"x{i}"]["kMeans"] = {
            "k": k
        }

    for x in clustering_results:
        dataframe = clustering_results[x]["dataframe"]
        k = clustering_results[x]["kMeans"]["k"]
        labels = kMeans.k_means(dataframe, k)
        clustering_results[x]["kMeans"]["labels"] = labels
        Visualization.visualize_clusters(dataframe, labels, title=f"Clusteringergebnis für {x} mit k={k} (k-Means)")