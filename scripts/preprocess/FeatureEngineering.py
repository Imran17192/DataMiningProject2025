import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FeatureEngineering:
    def __init__(self, dfs):
        self.dfs = dfs

    def normalize_df(self, dfs):
        dfs_norm = []
        for df in dfs:
            df_norm = (df - df.min()) / (df.max() - df.min())
            dfs_norm.append(df_norm)
        return dfs_norm


    def compute_features(self):
        for df in self.dfs:
            # Before we do PCa we have to center and scale the data(already done but double last better)
            scaled_data = preprocessing.scale(df.T)
            # We create a PCA object. rather than function that does PCA and returns results,
            # sklearn uses objects that n be trained using one das and applied to another
            pca = PCA()
            # does all pca math
            pca.fit(scaled_data)
            # coordinates for pca graph based on the loading scores and the sclaed data
            pca_data = pca.transform(scaled_data)
            # claclulate percentage of variation that each pc accounts for
            per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
            # create labels for screee plot
            labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

            plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
            plt.ylabel('Percentage of explained variance')
            plt.xlabel('Principal components')
            plt.title('scree plot')
            plt.show()

            pca_df = pd.DataFrame(pca_data,index=range(1,len(pca_data)+1), columns=labels)

            plt.scatter(pca_df.PC1, pca_df.PC2)
            plt.title('PCA')
            plt.xlabel("PC1-{0}%".format(per_var[0]))
            plt.xlabel("PC2-{0}%".format(per_var[1]))

            for sample in pca_df.index:
                plt.annotate(sample, (pca_df.PC1[sample], pca_df.PC2[sample]))

            plt.show()