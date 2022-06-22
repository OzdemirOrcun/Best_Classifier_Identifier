from utils.logger import logger
import warnings
import itertools

import numpy as np

from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import mutual_info_classif

from scipy.stats import pearsonr
from scipy import stats

import skfuzzy as fuzz

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


# StatisticalMetricsCalculator calculates desired statistical metrics of given generated dataset.

class StatisticalMetricsCalculator:
    def __init__(self, generated_dataframe=None,
                 generated_label_dataframe=None,
                 generated_dataset=None):

        self.df = generated_dataframe
        self.X = generated_dataset
        self.label_df = generated_label_dataframe
        self.statistical_metrics_dict = {}

        # in order to handle multiclass label data the imbalance ratio of the majority to the rest

    def calculate_inbalace_ratio(self):
        try:
            vc = self.label_df.value_counts()
            inbalance_ratio = sorted(vc)[-1] / sum(sorted(vc)[:-1])
            self.statistical_metrics_dict['inbalance_ratio'] = inbalance_ratio
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['inbalance_ratio'] = np.nan

        return None

    # Calculates Number of Features
    def calculate_number_features(self):
        try:
            number_features = len(self.df.columns)
            self.statistical_metrics_dict['number_features'] = number_features
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['number_features'] = np.nan

        return None

    # Calculates Number of Instances
    def calculate_number_instances(self):
        try:
            number_instances = len(self.df)
            self.statistical_metrics_dict['number_instances'] = number_instances
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['number_instances'] = np.nan
        return None

    # Calculates Number of Decision Classes
    def calculate_number_decision_classes(self):
        try:
            number_decision_classes = self.label_df.nunique()[0]
            self.statistical_metrics_dict['number_decision_classes'] = number_decision_classes
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['number_decision_classes'] = np.nan
        return None

    # Calculates Fuzzy Partition Coefficent based on rough set
    def calculate_fuzzy_partition_coefficient(self):
        try:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                self.X, c=self.statistical_metrics_dict['number_decision_classes'], m=2, error=0.005, maxiter=1000, seed=0, init=None)
            rounded_fpc = round(fpc, 2)
            self.statistical_metrics_dict['fpc'] = rounded_fpc
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['fpc'] = np.nan

        return None

    # Calculates overall standard deviation of dataframe.
    # Flattens the dataset and take the std of the row.
    def calculate_std_dataframe(self):
        try:
            std = np.std(np.array(self.df).reshape(1, self.df.shape[0] * self.df.shape[1]))
            self.statistical_metrics_dict['std_dataframe'] = std
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['std_dataframe'] = np.nan
        return None

    # Calculates Average Correlation Feature -> Class
    # Uses mutual_info_classif and takes mean the result.

    def calculate_average_correlation_feature_class(self):
        try:
            acfc = np.mean(mutual_info_classif(self.df, self.label_df))
            self.statistical_metrics_dict['av_corr_fc'] = acfc
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['av_corr_fc'] = np.nan
        return None

    # Calculates Average Correlation Feature -> Feature
    # itertools creates 2 ** number of features,
    # and append each pairs pearson's correlation into the correlations_list
    # then takes the mean of the appended list.
    def calculate_average_correlation_feature_feature(self):

        try:
            correlations_list = []
            for i in itertools.product(list(self.df.columns), repeat=2):
                corr, _ = pearsonr(self.df.iloc[i[0], :], self.df.iloc[i[1], :])
                correlations_list.append(corr)

            averaged = np.mean(np.array(correlations_list))
            self.statistical_metrics_dict['av_corr_ff'] = averaged

        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['av_corr_ff'] = np.nan
        return None

    # Each feature's normality is calculated based on the scipy documentation
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    # null hypothesis: x comes from a normal distribution

    def calculate_number_of_normal_features(self, alpha=0.05):
        try:
            counter = 0
            for _, i in enumerate(self.X.T):
                k2, p = stats.normaltest(i)
                if p < alpha:
                    continue  # we reject the null hypothesis
                else:
                    counter += 1  # we cannot reject the null hypothesis
            self.statistical_metrics_dict['number_normal_features'] = counter

        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['number_normal_features'] = np.nan

        return None

    # Each feature's variance is calculated and appened into the var_lst
    # Then, the average of the list is taken.
    def calculate_average_variance(self):
        try:
            var_lst = []
            for i in self.X.T:
                var_lst.append(np.var(i))
            av_var = np.mean(np.array(var_lst))
            self.statistical_metrics_dict['average_variance'] = av_var
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['average_variance'] = np.nan
        return None

    # Calculates Average Correlation Feature -> Feature
    # itertools creates 2 ** number of features,
    # and append each pairs pearson's correlation into the correlations_list
    # then takes the standard deviation of the appended list.

    def calculate_standard_deviation_correlation_feature_feature(self):

        try:
            correlations_list = []
            for i in itertools.product(list(self.df.columns), repeat=2):
                corr, _ = pearsonr(self.df.iloc[i[0], :], self.df.iloc[i[1], :])
                correlations_list.append(corr)

            standard_deviaton = np.std(np.array(correlations_list))
            self.statistical_metrics_dict['std_corr_ff'] = standard_deviaton
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['std_corr_ff'] = np.nan
        return None

    # Calculates Standard Deviation Correlation Feature -> Class
    # Uses mutual_info_classif and takes mean the result.

    def calculate_standard_deviation_correlation_feature_class(self):
        try:
            stdcfc = np.std(mutual_info_classif(self.df, self.label_df))

            self.statistical_metrics_dict['std_corr_fc'] = stdcfc
        except Exception as e:
            logger.error(e)
            self.statistical_metrics_dict['std_corr_fc'] = np.nan
        return None

    # Command control of calculations.
    def prepare_statistical_metrics_dict(self):
        self.calculate_inbalace_ratio()
        self.calculate_number_features()
        self.calculate_number_instances()
        self.calculate_number_decision_classes()
        self.calculate_fuzzy_partition_coefficient()
        self.calculate_std_dataframe()
        self.calculate_average_correlation_feature_class()
        self.calculate_average_correlation_feature_feature()
        self.calculate_number_of_normal_features()
        self.calculate_average_variance()
        self.calculate_standard_deviation_correlation_feature_feature()
        self.calculate_standard_deviation_correlation_feature_class()
        return None

    # Basic getter function for detailed examinations
    def get_statistical_metrics_dict(self):
        return self.statistical_metrics_dict
