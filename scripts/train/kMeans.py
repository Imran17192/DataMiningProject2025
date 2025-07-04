from sklearn.cluster import KMeans

from kneed import KneeLocator

class kMeans:

    def __init__(self, dataframes):
        self.dataframes = dataframes if type(dataframes)==list else [dataframes]

    @staticmethod
    def k_means(dataframe, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dataframe)
        return kmeans.labels_

    def elbow_analysis(self, k_max=30):
        dataframes_wcss = []
        for dataframe in self.dataframes:
            dataframe_wcss = []
            for k in range(1, k_max+1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(dataframe)
                dataframe_wcss.append((k, kmeans.inertia_))
            dataframes_wcss.append(dataframe_wcss)
        return dataframes_wcss

    @staticmethod
    def locate_knee(wcss):
        kl = KneeLocator([w[0] for w in wcss], [w[1] for w in wcss], curve="convex", direction="decreasing")
        return kl.elbow