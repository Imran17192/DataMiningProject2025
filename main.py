import json

from sklearn.model_selection import train_test_split

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

    df_ds1 = []
    for p in paths.DS1_DIR:
        df = pd.read_json(p)
        df_ds1.append(df)

    # DS2 not a pandas datframe so do it later. it is also just  for puzzle

    return df_x, df_ds1


def load_labels():
    labels = []
    for labels_path in [paths.Y0_DIR, paths.Y1_DIR, paths.Y2_DIR]:
        with open(labels_path) as labels_file:
            y = json.load(labels_file)
            labels.append(y)
    return labels


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
    kmean.run_k_means(opt_k=False, input_k=False, subplots=False, evaluate=False)
    dbscan.run_DBSCAN(fast_compute=False)
    for i, df in enumerate(df1):
        print(f"\n=== Erweiterte Clusterverfahren für df{i} ===")
        kmean.fuzzy_c_means(df, c=3)
        kmean.mini_batch_kmeans(df, k=3)
        kmean.em_gaussian_mixture(df, n_components=3)


def dm_part3(df_x, labels):
    # Verschmelzen
    dataframes_labeled = [dataframe.copy() for dataframe in df_x]
    for x, y in zip(dataframes_labeled, labels):
        x['y'] = y

    # Vorverarbeiten
    dataframes_labeled_preprocessed=[]
    for dataframe in dataframes_labeled:
        dataframe_cleaned = Preprocessing.remove_outliers_labels(dataframe, labels_column='y')
        x = dataframe_cleaned.drop(columns=['y'])
        y = dataframe_cleaned['y']
        dataframe_standardized = Preprocessing.standardize(x)
        dataframe_preprocessed = Preprocessing.principal_component_analysis(dataframe_standardized, main_components=3)
        dataframe_preprocessed['y'] = y.values
        dataframes_labeled_preprocessed.append(dataframe_preprocessed)

    # Split 70-30
    dataframes_train = []
    dataframes_test = []
    for dataframe in dataframes_labeled_preprocessed:
        x = dataframe.drop(columns=['y'])
        y = dataframe['y']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        dataframe_train = pd.concat([x_train, y_train], axis=1)
        dataframe_test = pd.concat([x_test, y_test], axis=1)
        dataframes_train.append(dataframe_train)
        dataframes_test.append(dataframe_test)


if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    #df_x_preprocessed, df_ds1_preprocessed = dm_part1(df_x, df_ds1)

    #dm_part2(df_x_preprocessed, df_ds1_preprocessed)

    labels_y = load_labels()

    dm_part3(df_x, labels_y)