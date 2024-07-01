import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import classification_report, RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

## Red wine == 1 ; White wine == 0

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter = int(1e6)),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'Support Vector Classifier': SVC(probability = True),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(eval_metric = 'mlogloss')
    }
    return models

model_params = {
    'Logistic Regression': {'C': list(range(-5, 6)), 'penalty': ['l1', 'l2', 'elasticnet', 'None'], 'solver': ['lbfgs', 'liblinear', 'saga']},
    'Gaussian Naive Bayes': {'var_smoothing': list(range(3, -4, -1))},
    'K-Nearest Neighbors': {'n_neighbors': list(range(1, 11)), 'leaf_size': list(range(1, 31, 5)), 'p': [1, 2], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']},
    'Linear Discriminant Analysis': {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': [i / 10 for i in range(0, 11, 2)] + ['auto', None],  'n_components': [i for i in range(8)], 'store_covariance': [False, True]},
    'Quadratic Discriminant Analysis': {'reg_param': [i / 10 for i in range(0, 11, 2)]},
    'Support Vector Classifier': {'C': np.logspace(-3, 3, num = 7, base = 10), 'kernel': ['rbf', 'sigmoid'], 'gamma': np.logspace(-3, 3, num = 7, base = 10)},
    'Random Forest': {'n_estimators': list(range(50, 501, 50)) + [1], 'max_depth': list(range(1, 12, 2)), 'max_features': ['sqrt', 'log2'], 'criterion': ['gini', 'entropy']},
    'XGBoost': {'max_depth': list(range(1, 12, 2)), 'n_estimators': list(range(50, 301, 50)) + [1]}
}
class ModelTuner:
    def __init__(self, model, params, x, y, cv) -> None:
        self.model = model
        self.params = params
        self.x = x
        self.y = y
        self.cv = cv
        self.grid_search = None

    def tune_model(self):
        self.grid_search = GridSearchCV(estimator = self.model, param_grid = self.params, cv = self.cv, n_jobs = -1, scoring = "accuracy")
        self.grid_search.fit(self.x, self.y)
        return self.grid_search.best_params_, self.grid_search.best_score_

class FeatureSelector:
    def __init__(self, cv, model, x, y) -> None:
        self.cv = cv
        self.model = model
        self.x = x
        self.y = y
        self.sfs = None

    def select_features(self):
        self.sfs = SFS(estimator = self.model, cv = self.cv, forward = True, floating = True, k_features = "best", scoring = "accuracy", n_jobs = -1)
        self.sfs.fit(self.x, self.y)
        return self.sfs.k_feature_names_

class WineTypeModel:
    def __init__(self, x_train, x_test, y_train, y_test, name) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.name = name
    
    def train_and_evaluate(self, model, params):
        if self.name == "Support Vector Classifier":
            cls_ = SVC(probability = True, **params)
        else:    
            cls_ = model(**params)
        cls_.fit(self.x_train, self.y_train)
        y_pred = cls_.predict(self.x_test)

        self._print_classification_report(y_pred)
        self._plot_roc_curve(cls_)
        self._plot_decision_region(cls_)
    
    def _print_classification_report(self, y_pred):
        print(classification_report(self.y_test, y_pred))
    
    def _plot_roc_curve(self, model):
        fpr, tpr, thresholds = roc_curve(self.y_test, model.predict_proba(self.x_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr = fpr, tpr = tpr, roc_auc = roc_auc, estimator_name = model.__class__.__name__
            )
        display.plot()
        plt.title(f"{self.name} ROC curve")
        plt.savefig(f"result/{self.name} ROC curve", dpi = 300)
        # plt.show()
        plt.close()

    def _plot_decision_region(self, model):
        pca = PCA(n_components = 2)
        x_train_pca = pca.fit_transform(self.x_train)
        x_test_pca = pca.transform(self.x_test)
        model_fit_pca = model.fit(x_train_pca, self.y_train)
        
        plot_decision_regions(x_test_pca, self.y_test.values, clf = model_fit_pca, colors = '#fffacd,#a00028')
        plt.title(f"{self.name} Decision Region")
        plt.savefig(f"result/{self.name} Decision Region", dpi = 300)
        # plt.show()
        plt.close()
    
def main():
    warnings.filterwarnings('ignore')
    os.chdir("/Users/shenchingfeng/GitHub/ML-Wine-Type-and-Quality-Classification/")
    df = pd.read_csv("data/Wine.csv")
    kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2024)
    
    models = get_models()
    model_name, model_score, model_features, tuned_params = [], [], [], []

    for name, model in models.items():
        print(f"Tuning Parameters {name}...")

        x = df.drop(columns = ["type", "quality", "alcohol", "pH", "fixed acidity"])
        y = df["type"]

        if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Classifier']:
            x = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 2024)
        
        param = model_params[name]
        tuner = ModelTuner(model = model, params = param, x = x, y = y, cv = kf)
        best_params, score = tuner.tune_model()

        print(f"Selecting Features {name}...")
        selector = FeatureSelector(cv = kf, model = model, x = x, y = y)
        best_features = selector.select_features()

        print(f"Training {name}...")
        trainer = WineTypeModel(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, name = name)
        trainer.train_and_evaluate(model = type(model), params = best_params)

        model_name.append(name)
        model_score.append(round(score, 4))
        model_features.append(best_features)
        tuned_params.append(best_params)

    model_tune_result = pd.DataFrame({
        "Model Name": model_name, "Model Score": model_score, "Model Features": model_features, "Model Params": tuned_params
    })
    
    model_tune_result.to_csv("result/Result.csv", index = False)

if __name__ == "__main__":
    main()