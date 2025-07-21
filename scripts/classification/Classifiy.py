import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import Axes3D


class Classify:
    def __init__(self, X_train, y_train, X_valid=None, y_valid=None, model_type="svm"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.y_test_pred = None

        self.model_type = model_type
        self.model = None

        self.choose_model(model_type)


    def choose_model(self, model_type):
        if model_type == "svm":
            self.model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
        elif model_type == "logreg":
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
        elif model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        elif model_type == "gnb":
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")

    def train_svm(self, plot=False):
        self.model.fit(self.X_train, self.y_train)
        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            print(f"[SVM] Accuracy: {acc:.4f}")

            if plot:
                self.visualize_confusion_matrix(self.y_valid, y_pred, "SVM Confusion")
                self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred, title="SVM Prediction")
                self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred, title="SVM Prediction 3d")
                self.plot_svm_decision_boundary()

        self.y_test_pred = self.model.predict(self.X_train)
        if plot:
            self.visualize_prediction_embedding(self.X_train, self.y_train, self.y_test_pred,
                                                title="SVM Test-Prediction")
            self.visualize_prediction_embedding_3d(self.X_train, self.y_train, self.y_test_pred,
                                                   title="SVM Test-Prediction 3D")
        self.save_predictions(self.y_test_pred, path="predictions/svm_prediction.json")

    def train_logreg(self, plot=False):
        if not isinstance(self.model, LogisticRegression):
            raise TypeError("Nur für LogisticRegression verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            loss = log_loss(self.y_valid, self.model.predict_proba(self.X_valid))
            print(f"[LOGREG] Accuracy: {acc:.4f} | LogLoss: {loss:.4f}")

            if plot:
                self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – Logistic Regression")
                self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred, title="LogReg Prediction")
                self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred, title="LogReg Prediction 3d")

        self.y_test_pred = self.model.predict(self.X_train)
        if plot:
            self.visualize_prediction_embedding(self.X_train, self.y_train, self.y_test_pred,
                                                title="LogReg Test-Prediction")
            self.visualize_prediction_embedding_3d(self.X_train, self.y_train, self.y_test_pred,
                                                   title="LogReg Test-Prediction 3D")
        self.save_predictions(self.y_test_pred, path="predictions/logreg_prediction.json")

    def train_gnb(self, plot=False):
        if not isinstance(self.model, GaussianNB):
            raise TypeError("Nur für GaussianNB verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            print(f"[GNB] Accuracy: {acc:.4f}")

            if plot:
                self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – Naive Bayes")
                self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred, title="GNB Prediction")
                self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred, title="GNB Prediction 3d")

        self.y_test_pred = self.model.predict(self.X_train)
        if plot:
            self.visualize_prediction_embedding(self.X_train, self.y_train, self.y_test_pred,
                                                title="GNB Test-Prediction")
            self.visualize_prediction_embedding_3d(self.X_train, self.y_train, self.y_test_pred,
                                                   title="GNB Test-Prediction 3D")
        self.save_predictions(self.y_test_pred, path="predictions/gnb_prediction.json")

    def train_knn(self, plot=False):
        if not isinstance(self.model, KNeighborsClassifier):
            raise TypeError("Nur für KNeighborsClassifier verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred, average="weighted")
            print(f"[KNN] Accuracy: {acc:.4f} ")

            if plot:
                self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – k-NN")
                self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred, title="KNN Prediction")
                self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred, title="KNN Prediction 3d")
                self.plot_knn_decision_boundary()

        self.y_test_pred = self.model.predict(self.X_train)
        if plot:
            self.visualize_prediction_embedding(self.X_train, self.y_train, self.y_test_pred,
                                                title="KNN Test-Prediction")
            self.visualize_prediction_embedding_3d(self.X_train, self.y_train, self.y_test_pred,
                                                   title="KNN Test-Prediction 3D")
        self.save_predictions(self.y_test_pred, path="predictions/knn_prediction.json")

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def save_predictions(y_pred, path="predictions.json"):
        with open(path, "w") as f:
            json.dump(y_pred.tolist(), f)

    def visualize_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Vorhergesagt")
        plt.ylabel("Tatsächlich")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_prediction_embedding(X, y_true, y_pred, method="tsne", title="_"):
        embed = TSNE(n_components=2, random_state=42).fit_transform(X)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(embed[:, 0], embed[:, 1], c=y_true, cmap="tab20", s=5)
        plt.title("True Labels")
        plt.subplot(1, 2, 2)
        plt.scatter(embed[:, 0], embed[:, 1], c=y_pred, cmap="tab20", s=5)
        plt.title("Predicted Labels")
        plt.suptitle(f"TSNE Projektion – Wahr vs. Vorhergesagt – {title}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_prediction_embedding_3d(X, y_true, y_pred, method="pca", title="_"):
        if method == "pca":
            pca = PCA(n_components=3)
            embed = pca.fit_transform(X)
        else:
            raise ValueError("Nur PCA für 3D unterstützt")

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c=y_true, cmap='tab20', s=10)
        ax1.set_title("True Labels (3D PCA)")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c=y_pred, cmap='tab20', s=10)
        ax2.set_title("Predicted Labels (3D PCA)")

        plt.suptitle(f"3D PCA-Projektion – Wahr vs. Vorhergesagt – {title}")
        plt.tight_layout()
        plt.show()

    def plot_svm_decision_boundary(self, title="SVM Decision Boundary (PCA)"):
        if not isinstance(self.model, SVC):
            return
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.X_train)
        model_2d = SVC(kernel="rbf", C=10, gamma=0.01)
        model_2d.fit(X_reduced, self.y_train)
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab20")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.y_train, cmap="tab20", s=15, edgecolors="k")
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    def plot_knn_decision_boundary(self, title="k-NN Decision Boundary (PCA)"):
        if not isinstance(self.model, KNeighborsClassifier):
            return

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.X_train)

        model_2d = KNeighborsClassifier(n_neighbors=5)
        model_2d.fit(X_reduced, self.y_train)

        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))

        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab20")
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.y_train, cmap="tab20", s=15, edgecolors="k")
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()
