from utils.logger import logger
import warnings
import itertools

import pandas as pd
import numpy as np

from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


# MetaDatasetGenerator predicts generated dataset to decide which machine learning model is the most suitable,
# and combines the results from the StatisticalMetricsCalculator with the metadataset label

class MetaDatasetGenerator:

    def __init__(self, generated_dataframe=None,
                 generated_label_dataframe=None,
                 statistical_metrics=None):

        self.generated_dataset = generated_dataframe
        self.generated_label = generated_label_dataframe
        self.statistical_metrics = statistical_metrics

        self.metadata_dict = {}

        # Initial models
        self.model_dict = {'SVM': SVC(), 'RFC': RandomForestClassifier()}

        self.rf_hp_dict = {
            "criterion": ['gini', 'entropy'],
            "max_features": [None, "sqrt", "log2"],
            "n_estimators": [100, 500, 1000]
        }

        self.svm_hp_dict = {
            "gamma": ['scale', 'auto'],
            "kernel": ["poly", "rbf", "sigmoid"],
            "C": [1, 10, 100]
        }

        self.metadata_label_matrix = None
        self.label_dict = None
        self.rfc_label_dict = None
        self.svm_label_dict = None

        self.cv = 5

        self.hyperparameters_dict = {"SVM": self.svm_hp_dict, "RFC": self.rf_hp_dict}

        self.score_dict = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.winner_model = None
        self.winner_score = None
        self.winner_hp = None

    def create_multi_label_dict(self):
        gamma = self.svm_hp_dict['gamma']
        kernel = self.svm_hp_dict['kernel']
        C = tuple(self.svm_hp_dict['C'])

        criterion = self.rf_hp_dict['criterion']
        max_feature = self.rf_hp_dict['max_features']
        estimators = self.rf_hp_dict['n_estimators']

        RF = [("RFC",) + i for i in list(itertools.product(*[criterion, max_feature, estimators]))]

        SVM = [("SVM",) + i for i in list(itertools.product(*[gamma, kernel, C]))]

        RF_set = [frozenset(i) for i in RF]
        SVM_set = [frozenset(i) for i in SVM]

        self.rfc_label_dict = dict(zip(RF_set, [i for i in range(18)]))
        self.svm_label_dict = dict(zip(SVM_set, [i for i in range(18, 36)]))

        self.rfc_label_dict.update(self.svm_label_dict)
        self.label_dict = self.rfc_label_dict

        return None

    # Train test split operation
    def split_data(self, ratio=0.8, random_state=0):

        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.generated_dataset,
                                                                                    self.generated_label,
                                                                                    train_size=ratio,
                                                                                    random_state=random_state)
        except Exception as e:
            logger.debug(e)

        return None

    # GridSearchs for given models and stores their score, model name, model hyperparameters
    def hp_tuning_and_score_data(self):

        try:
            for k, v in self.model_dict.items():

                grid = GridSearchCV(v, self.hyperparameters_dict[k], refit=True, cv=self.cv, verbose=0)
                grid.fit(self.generated_dataset, self.generated_label)

                def get_top_hyperparameters(grid):
                    df = pd.DataFrame(grid.cv_results_["params"])
                    df['accuracy'] = np.round(grid.cv_results_["mean_test_score"], 3)
                    df.sort_values(by='accuracy', ascending=False, inplace=True)
                    top_hps_acc = df[np.array(df.head(1)['accuracy']) - df['accuracy'] <= 0.02]

                    top_hps = top_hps_acc.iloc[:, :-1].values
                    top_acc = top_hps_acc.iloc[:, -1].values

                    top_hps = [tuple(i) for i in top_hps]
                    top_acc = [i for i in top_acc]

                    return top_hps, top_acc

                top_hps, top_acc = get_top_hyperparameters(grid)

                for i in zip(top_hps, top_acc):
                    model_with_h_parameters = frozenset(set((k,) + i[0]))
                    self.score_dict[model_with_h_parameters] = i[1]

            self.winner_score = self.score_dict.values()

            # model_with_h_parameters = (k,set(tuple(grid.best_params_.values())))

            # self.score_dict[model_with_h_parameters] = round(grid.best_score_,3)

        except Exception as e:
            logger.error(e)

        return None

    def create_metadata_multi_label_matrix(self):

        self.hp_tuning_and_score_data()
        self.create_multi_label_dict()

        multi_label_matrix = np.zeros((1, 36))

        try:
        # index
            for hp, s in self.score_dict.items():
                print(self.label_dict)
                array_index = self.label_dict[hp]
                multi_label_matrix[0][array_index] = 1

            self.metadata_label_matrix = multi_label_matrix

        except Exception as e:
            print(self.score_dict)
            logger.error(e)

        return None

    # Model label and statistical measure combination.
    def generate_metadata_dict(self):
        # self.generate_meta_label()

        self.create_metadata_multi_label_matrix()
        self.metadata_dict = self.statistical_metrics

        # self.metadata_dict['Model'] = self.winner_model
        # self.metadata_dict.update(self.winner_hp)

        return None

    # Basic getter function for detailed examinations
    def get_winner_model(self):
        return self.winner_model

    # Basic getter function for detailed examinations
    def get_winner_score(self):
        return self.winner_score

    # Basic getter function for detailed examinations
    def get_score_dict(self):
        return self.score_dict

    # Basic getter function for detailed examinations
    def get_metadata_dict(self):
        return self.metadata_dict

    # Basic getter function for detailed examinations
    def get_winner_hp(self):
        return self.winner_hp

    def get_metadata_multi_label_matrix(self):
        return self.metadata_label_matrix

    def get_label_dict(self):
        return self.label_dict
