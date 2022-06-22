from utils.logger import logger
import random
import warnings

import pandas as pd
import numpy as np

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


# DatasetGenerator object creates dataframes according to randomly chosen features from features_dict
class DatasetGenerator:

    def __init__(self, features_dict=None):

        self.features_dict = features_dict
        self.selected_features = {}
        self.data_frame = None
        self.label_data_frame = None

        self.X = None
        self.y = None

    # In our initial tests we did not observed n_classes other than 2.
    # Therefore, we added bias to random choice to get different n_classes.
    # For the weights feature, np.random couldn't accept lists of list so we used the embedded random library.
    def select_features_random(self):
        selected_features = {}
        for k, v in self.features_dict.items():
            if k == 'n_classes':
                selected_features[k] = np.random.choice(v, p=[0.2, 0.5, 0.3])
            elif k == 'weights':
                selected_features[k] = random.choice(v)
            elif k == "n_informative":
                selected_features[k] = np.random.choice(v, p=[0.1, 0.2, 0.7])
            else:
                selected_features[k] = np.random.choice(v)
        return selected_features

    # Core of the dataframe generation operation.
    # To observe further, we keep the chosen features.
    def generate_dataset(self):
        self.selected_features = self.select_features_random()

        try:
            self.X, self.y = make_classification(**self.selected_features)
            # min-max scaler
            scaler = MinMaxScaler()
            scaler.fit(self.X)
            self.X = scaler.transform(self.X)

        except Exception as e:
            logger.debug(e)

        return self.X, self.y, self.selected_features

    # To make further operations more convenient, a converter function is written.
    def convert_dataset_to_dataframe(self):
        try:
            self.data_frame = pd.DataFrame(self.X)
            self.label_data_frame = pd.DataFrame(self.y)

        except Exception as e:
            logger.debug(e)

        return self.data_frame, self.label_data_frame, self.selected_features
