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
from mpl_toolkits.mplot3d import Axes3D  # wichtig für 3D-Plots



class Classify:
    def __init__(self, X_train, y_train, X_valid=None, y_valid=None, model_type="svm"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.model_type = model_type
        self.model = None

        self.choose_model(model_type)


    #TODO Remove logreg keep the rest
    #TODO edit code so you of classify
    def choose_model(self, model_type):
        if model_type == "svm":
            self.model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
        elif model_type == "logreg":
            self.model = LogisticRegression(
                multi_class='multinomial', solver='lbfgs',
                max_iter=500, random_state=42
            )
        elif model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42
            )
        elif model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        elif model_type == "gnb":
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")



    def predict(self, X):
        return self.model.predict(X)


    def save_predictions(self, y_pred, path="predictions.json"):
        with open(path, "w") as f:
            json.dump(y_pred.tolist(), f)


    def visualize_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Vorhergesagt")
        plt.ylabel("Tatsächlich")
        plt.tight_layout()
        plt.show()


    def visualize_prediction_embedding(self, X, y_true, y_pred, method="tsne"):
        embed = TSNE(n_components=2, random_state=42).fit_transform(X)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(embed[:, 0], embed[:, 1], c=y_true, cmap="tab20", s=5)
        plt.title("True Labels")
        plt.subplot(1, 2, 2)
        plt.scatter(embed[:, 0], embed[:, 1], c=y_pred, cmap="tab20", s=5)
        plt.title("Predicted Labels")
        plt.suptitle("TSNE Projektion – Wahr vs. Vorhergesagt")
        plt.tight_layout()
        plt.show()

    def visualize_prediction_embedding_3d(self, X, y_true, y_pred, method="pca"):
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

        plt.suptitle("3D PCA-Projektion – Wahr vs. Vorhergesagt")
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


    def train_svm(self):
        self.model.fit(self.X_train, self.y_train)
        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            print(f"[SVM] Accuracy: {acc:.4f}")

            self.visualize_confusion_matrix(self.y_valid, y_pred)
            self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred)
            self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred)

            self.plot_svm_decision_boundary()

        y_test_pred = self.model.predict(self.X_train)
        self.save_predictions(y_test_pred, path="predictions/svm_prediction.json")


    def train_logreg(self):
        if not isinstance(self.model, LogisticRegression):
            raise TypeError("Nur für LogisticRegression verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            loss = log_loss(self.y_valid, self.model.predict_proba(self.X_valid))
            print(f"[LOGREG] Accuracy: {acc:.4f} | LogLoss: {loss:.4f}")

            self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – Logistic Regression")
            self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred)
            self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred)

        y_test_pred = self.model.predict(self.X_train)
        self.save_predictions(y_test_pred, path="predictions/logreg_prediction.json")


    def train_rf(self):
        if not isinstance(self.model, RandomForestClassifier):
            raise TypeError("Nur für RandomForestClassifier verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred, average="weighted")
            print(f"[RF] Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

            self.plot_rf_prediction_vs_true()
            self.plot_rf_decision_boundary(title="Random Forest")
            self.plot_rf_feature_importance()
            self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – Random Forest")
            self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred)
            self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred)

        y_test_pred = self.model.predict(self.X_train)
        self.save_predictions(y_test_pred, path="predictions/rf_prediction.json")


    def plot_rf_prediction_vs_true(self):
        if self.X_valid is None or self.y_valid is None:
            return

        y_pred = self.model.predict(self.X_valid)

        plt.figure(figsize=(12, 6))
        plt.scatter(self.y_valid, y_pred, alpha=0.5)
        plt.xlabel("Wahr")
        plt.ylabel("Vorhergesagt")
        plt.title("Random Forest – True vs. Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_rf_decision_boundary(self, title="Random Forest Decision Boundary (PCA)"):
        if not isinstance(self.model, RandomForestClassifier):
            raise TypeError("Nur für RandomForestClassifier verfügbar.")

        # Auf 2D reduzieren
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.X_train)

        model_2d = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
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


    def plot_rf_feature_importance(self, feature_names=None):
        if not isinstance(self.model, RandomForestClassifier):
            raise TypeError("Nur für RandomForestClassifier verfügbar.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        plt.title("Feature Importance (Random Forest)")
        names = feature_names if feature_names else [f"Feature {i}" for i in range(len(importances))]
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [names[i] for i in indices], rotation=45)
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

    def train_knn(self):
        if not isinstance(self.model, KNeighborsClassifier):
            raise TypeError("Nur für KNeighborsClassifier verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred, average="weighted")
            print(f"[KNN] Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

            self.plot_knn_decision_boundary()
            self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – k-NN")
            self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred)
            self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred)

        y_test_pred = self.model.predict(self.X_train)
        self.save_predictions(y_test_pred, path="predictions/knn_prediction.json")

    def train_gnb(self):
        if not isinstance(self.model, GaussianNB):
            raise TypeError("Nur für GaussianNB verfügbar.")

        self.model.fit(self.X_train, self.y_train)

        if self.X_valid is not None and self.y_valid is not None:
            y_pred = self.model.predict(self.X_valid)
            acc = accuracy_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred, average="weighted")
            print(f"[GNB] Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

            self.visualize_confusion_matrix(self.y_valid, y_pred, title="Confusion Matrix – Naive Bayes")
            self.visualize_prediction_embedding(self.X_valid, self.y_valid, y_pred)
            self.visualize_prediction_embedding_3d(self.X_valid, self.y_valid, y_pred)

        y_test_pred = self.model.predict(self.X_train)
        self.save_predictions(y_test_pred, path="predictions/gnb_prediction.json")

