import json

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class Classify:

    def __init__(self, X_train, y_train, X_valid=None, y_valid=None, model_type="svm"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model_type = model_type

        self.model = self._init_model(model_type)

        self.model.fit(X_train, y_train)

        if X_valid is not None and y_valid is not None:
            self.evaluate()

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    def _init_model(self, model_type):
        if model_type == "svm":
            return SVC(kernel="rbf", C=10, gamma=0.01)
        elif model_type == "tree":
            return DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=42)
        elif model_type == "gnb":
            return GaussianNB()
        elif model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=300,
                early_stopping=True,
                random_state=42,
                learning_rate_init=0.001,
                solver="adam",
                activation="relu"
            )
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=3,
                random_state=42
            )
        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski"
            )
        elif model_type == "logreg":
            return LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=300,
                random_state=42
            )
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")

    def evaluate(self):
        y_pred = self.model.predict(self.X_valid)
        acc = accuracy_score(self.y_valid, y_pred)
        f1 = f1_score(self.y_valid, y_pred, average="weighted")
        print(f"[{self.model_type.upper()}] Accuracy: {acc:.4f}  |  F1-Score: {f1:.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def save_predictions(self, y_pred, path="predictions.json"):
        with open(path, "w") as f:
            json.dump(y_pred.tolist(), f)
