import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter = int(1e7)),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'Support Vector Classifier': SVC(max_iter = int(1e7), probability = False, verbose = 1),
        'Random Forest': RandomForestClassifier(),
        # 'XGBoost': XGBClassifier(eval_metric = 'mlogloss')
    }
    return models
    
model_params = {
    'Logistic Regression': {'C': list(range(-5, 6)), 'penalty': ['l1', 'l2', 'elasticnet', 'None'], 'solver': ['lbfgs', 'liblinear', 'saga']},
    'Gaussian Naive Bayes': {'var_smoothing': np.logspace(-15, 0, num = 16, base = 10)},
    'K-Nearest Neighbors': {'n_neighbors': list(range(1, 11)), 'leaf_size': list(range(1, 31, 5)), 'p': [1, 2], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']},
    'Linear Discriminant Analysis': {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': [i / 10 for i in range(0, 11, 2)] + ['auto', None],  'n_components': [i for i in range(8)], 'store_covariance': [False, True]},
    'Quadratic Discriminant Analysis': {'reg_param': [i / 10 for i in range(0, 11, 2)]},
    'Support Vector Classifier': {'C': np.logspace(-3, 3, num = 7, base = 10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': np.logspace(-3, 3, num = 7, base = 10)},
    'Random Forest': {'n_estimators': list(range(50, 501, 50)) + [1], 'max_depth': list(range(1, 12, 2)), 'max_features': ['sqrt', 'log2'], 'criterion': ['gini', 'entropy']},
    'XGBoost': {'max_depth': list(range(1, 12, 2)), 'n_estimators': list(range(50, 301, 50)) + [1], 'learning_rate': [round(float(x), 2) for x in np.linspace(start = 0.01, stop = 0.2, num = 10)]}
}

class Upsampler():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def apply_smote(self):
        smote = SMOTE(sampling_strategy = "all", random_state = 2024, n_jobs = -1, k_neighbors = 4)
        x_smote, y_smote = smote.fit_resample(self.x, self.y)
        return x_smote, y_smote

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
    def __init__(self, cv, model, params, x, y) -> None:
        self.cv = cv
        self.model = model
        self.params = params
        self.x = x
        self.y = y
        self.sfs = None

    def select_features(self):
        self.sfs = SFS(estimator = self.model(**self.params), cv = self.cv, forward = True, floating = True, k_features = "best", scoring = "accuracy", n_jobs = -1)
        self.sfs.fit(self.x, self.y)
        return self.sfs.k_feature_names_

class WineTypeModel:
    def __init__(self, x_train, x_test, y_train, y_test, name, r) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.name = name
        self.r = r
    
    def train_and_evaluate(self, model, params):
        if self.name == "Support Vector Classifier":
            # print("Identify SVC")
            cls_ = SVC(probability = True, max_iter = int(1e7), **params)
        else:    
            cls_ = model(**params)
        cls_.fit(self.x_train, self.y_train)
        y_pred = cls_.predict(self.x_test)

        self._print_classification_report(y_pred)
        self._plot_roc_curve(cls_)
        self._plot_confusion_matrix(cls_, y_pred)
    
    def _print_classification_report(self, y_pred):
        report = classification_report(self.y_test, y_pred, output_dict = True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"result/Wine_Quality/{self.r}/{self.name} Classification Report.csv")
    
    def _plot_roc_curve(self, model):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        class_name = []
        n_classes = len(set(self.y_test))
        y_test_binarized = label_binarize(self.y_test, classes = model.classes_)
        y_score = model.predict_proba(self.x_test)

        for i in range(len(model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            class_name.append(i)
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        for i in class_name:
            plt.plot(
                fpr[i], tpr[i], linewidth = 2, label = f"class {i + 3}"
            )

        plt.plot(fpr["micro"], tpr["micro"], linewidth = 2, label = f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw = 2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{self.name} ROC curve_{self.r}")
        plt.legend(loc = 'lower right')
        plt.savefig(f"result/Wine_Quality/{self.r}/{self.name} ROC curve", dpi = 300)
        # plt.show()
        plt.close()

    def _plot_confusion_matrix(self, model, y_pred):
        cm = metrics.confusion_matrix(y_true = self.y_test, y_pred = y_pred, labels = model.classes_)

        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
        disp.plot(colorbar = "bwr")
        plt.title(f"{self.name} Confusion Matrix_{self.r}")
        plt.savefig(f"result/Wine_Quality/{self.r}/{self.name} Confusion Matrix", dpi = 300)
    
def main():
    warnings.filterwarnings('ignore')
    os.chdir("/Users/shenchingfeng/GitHub/ML-Wine-Type-and-Quality-Classification/")
    df = pd.read_csv("data/Wine.csv")
    kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 2024)
    training_round = ["Original data", "Upsample data"]
    # training_round = ["Upsample data"]

    for r in training_round:

        models = get_models()
        model_name, model_score, model_features, tuned_params, cal_time = [], [], [], [], []

        for name, model in models.items():

            print(f"Tuning Parameters {name} / {r}...")

            x = df.drop(columns = ["quality", "alcohol", "pH", "fixed acidity"])
            y = df["quality"]

            if r == "Upsample data":
                if name == "Logistic Regression":
                    print("===== Switch to Upsample data =====")

                    print(f"Original shape x: {x.shape}")
                    print(f"Original shape y: {y.shape}")
                
                x, y = Upsampler(x, y).apply_smote()

                if name == "Logistic Regression":
                    print("===== After Upsample =====")

                    print(f"Upsample shape x: {x.shape}")
                    print(f"Upsample shape y: {y.shape}")

            if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Classifier']:
                x = StandardScaler().fit_transform(x)
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 2024)

            if name != "Support Vector Classifier":
                param = model_params[name]
                tuner = ModelTuner(model = model, params = param, x = x, y = y, cv = kf)
                best_params, score = tuner.tune_model()

            else:
                best_score_svc = -1
                best_params_svc = {}

                for c in tqdm(np.logspace(-3, 3, num = 7, base = 10)):
                    for k in tqdm(['linear', 'poly', 'rbf', 'sigmoid']):
                        for g in tqdm(np.logspace(-3, 3, num = 7, base = 10)):

                            m = SVC(C = c, kernel = k, gamma = g)
                            m.fit(x_train, y_train)
                            y_pred_svc = m.predict(x_test)
                            score_svc = accuracy_score(y_test, y_pred_svc)
                            if score_svc > best_score_svc:
                                best_score_svc = score_svc
                                best_params_svc["C"] = c
                                best_params_svc["gamma"] = g
                                best_params_svc["kernel"] = k

                    gc.collect()

            print(f"Selecting Features {name} / {r}...")
            if name != "Support Vector Classifier":
                selector = FeatureSelector(cv = kf, model = type(model), params = best_params, x = x, y = y)
            else:
                selector = FeatureSelector(cv = kf, model = SVC(C = best_params_svc["C"], kernel = best_params_svc["kernel"], gamma = best_params_svc["gamma"], probability = False, max_iter = int(1e7)))
            best_features = selector.select_features()

            start_time = time.time()

            print(f"Training {name} / {r}...")
            trainer = WineTypeModel(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, name = name, r = r)
            trainer.train_and_evaluate(model = type(model), params = best_params)

            end_time = time.time()
            total_time = end_time - start_time

            model_name.append(name)
            model_score.append(round(score, 4))
            model_features.append(best_features)
            tuned_params.append(best_params)
            cal_time.append(round(total_time, 2))

            gc.collect()

        model_tune_result = pd.DataFrame({
            "Model Name": model_name, "Model Score": model_score, "Model Features": model_features, "Model Params": tuned_params, "Training Times (sec)": cal_time
        })
        
        model_tune_result.to_csv(f"result/Wine_Quality/{r}/Result.csv", index = False)

if __name__ == "__main__":
    main()