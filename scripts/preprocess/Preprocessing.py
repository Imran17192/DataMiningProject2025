import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Preprocessing:

    @staticmethod
    def preprocess(dataframe, main_components=2):
        dataframe_cleaned = Preprocessing.remove_outliers(dataframe)
        dataframe_standardized = Preprocessing.standardize(dataframe_cleaned)
        dataframe_preprocessed = Preprocessing.principal_component_analysis(dataframe_standardized, main_components)
        return dataframe_preprocessed

    @staticmethod
    def remove_outliers(dataframe):
        # Berechne IQR (Interquartilsabstand)
        Q1 = dataframe.quantile(1 / 4)
        Q3 = dataframe.quantile(3 / 4)
        IQR = Q3 - Q1

        # Entferne Ausreißer, die außerhalb des Bereichs [Q1 - 1.5*IQR, Q3 + 1.5*IQR] liegen
        dataframe_cleaned = dataframe[~((dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR))).any(axis=1)]

        return dataframe_cleaned

    @staticmethod
    def standardize(dataframe):
        # Standardisierung der Daten
        scaler = StandardScaler()
        dataframe_standardized = pd.DataFrame(scaler.fit_transform(dataframe))

        return dataframe_standardized

    @staticmethod
    def principal_component_analysis(dataframe, main_components=2):
        # Hauptkomponentenanalyse (PCA)
        pca = PCA(n_components=main_components)
        dataframe_pca = pd.DataFrame(pca.fit_transform(dataframe))

        return dataframe_pca