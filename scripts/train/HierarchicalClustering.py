from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

class HierarchicalClustering:

    def __init__(self, dataframes):
        self.dataframes = dataframes if type(dataframes)==list else [dataframes]
        self.labels = None

    def silhouette_analysis(self, linkage_method, c_max=30):
        print(f"Silhouettenanalyse für {linkage_method}-Linkage-Verfahren:")
        dataframes_silhouette_scores = []
        for i, dataframe_sample in enumerate(self.__sample_dataframes()):
            print(f"\tx{i}:")
            dataframe_silhouette_scores = []
            for c in range(2, c_max+1):
                score = HierarchicalClustering.__calculate_silhouette_score(dataframe_sample, c, linkage_method)
                print(f"\t\tc={c}: {score}")
                dataframe_silhouette_scores.append((c, score))
            dataframes_silhouette_scores.append(dataframe_silhouette_scores)
        return dataframes_silhouette_scores

    def __sample_dataframes(self):
        dataframes_samples = []
        for dataframe in self.dataframes:
            if dataframe.shape[0] > 10000:
                dataframes_samples.append(dataframe.sample(frac=0.1, random_state=42))
            else:
                dataframes_samples.append(dataframe)
        return dataframes_samples

    @staticmethod
    def __calculate_silhouette_score(dataframe, c, linkage_method):
        clustering = AgglomerativeClustering(n_clusters=c, linkage=linkage_method)
        labels = clustering.fit_predict(dataframe)
        score = silhouette_score(dataframe, labels)
        return score

    @staticmethod
    def linkage_clustering(dataframe, linkage_method, c):
        model = AgglomerativeClustering(n_clusters=c, linkage=linkage_method)
        return model.fit_predict(dataframe)